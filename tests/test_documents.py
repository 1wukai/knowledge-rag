import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from interview_rag.document_processing import DocumentParseError, chunk_document, decode_text, extract_sections, MAX_SOURCE_BYTES
from interview_rag.manifest import Manifest, ManifestDocument, add_documents, file_sha256, scan_candidates, save_manifest, refresh_document
from interview_rag.application.ingest import ingest_documents
from interview_rag.application.snapshot import freeze_documents
from interview_rag.application.build import build_generation
from interview_rag.config import Settings
from interview_rag.runtime_settings import defaults
from interview_rag.storage.vectorstore import store_for_settings, _stores

PROJECT = Path(__file__).resolve().parents[1]


def pdf_bytes(texts=("PDF test evidence",)):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=500, height=500)
        if text is None:
            continue
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type1'), NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 450 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    result = BytesIO()
    writer.write(result)
    return result.getvalue()


def docx_bytes():
    document = Document()
    document.add_paragraph('Word 第一段')
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = '预约口令'
    table.cell(0, 1).text = '松果七号'
    document.add_paragraph('Word 最后段')
    result = BytesIO()
    document.save(result)
    return result.getvalue()


def chunks(raw, path='test.txt', max_chars=1200):
    return chunk_document(raw, doc_id='test-doc', title='测试', path=path, source_hash='sha256:' + hashlib.sha256(raw).hexdigest(), max_chars=max_chars)


class ExtractionTests(unittest.TestCase):
    def test_text_encodings(self):
        for encoding in ('utf-8', 'utf-8-sig', 'utf-16', 'gb18030'):
            self.assertEqual(decode_text('三国演义\n桃园结义'.encode(encoding)), '三国演义\n桃园结义')
        with self.assertRaises(DocumentParseError): decode_text(b'abc\x00def')

    def test_plain_text_does_not_parse_markdown_and_bounds_long_lines(self):
        text = '---\n# 这不是标题\n' + '中' * 4100
        result = chunks(text.encode(), max_chars=200)
        self.assertTrue(all(len(c.text) <= 200 for c in result))
        self.assertEqual(len({c.chunk_id for c in result}), len(result))
        self.assertIn('---', result[0].text)
        self.assertIn('# 这不是标题', result[0].text)
        self.assertTrue(all(c.start_line == 3 and c.end_line == 3 for c in result[1:]))

    def test_pdf_pages_and_word_table_order(self):
        pdf = chunks(pdf_bytes(('Page one evidence', 'Page two evidence')), 'test.PDF')
        self.assertEqual(len(pdf), 2)
        self.assertIn('第 2 页', pdf[1].heading[0])
        self.assertIn('Page two', pdf[1].text)
        word = chunks(docx_bytes(), 'test.docx')
        text = '\n'.join(c.text for c in word)
        self.assertLess(text.index('第一段'), text.index('预约口令'))
        self.assertLess(text.index('松果七号'), text.index('最后段'))
        self.assertIn('提取文本', word[0].heading[0])

    def test_invalid_encrypted_and_scanned_documents(self):
        for path, raw in [('test.pdf', b'invalid'), ('test.docx', b'invalid'), ('test.doc', b'invalid'), ('test.txt', b'   ')]:
            with self.subTest(path=path), self.assertRaises(DocumentParseError): chunks(raw, path)
        with self.assertRaisesRegex(DocumentParseError, 'OCR'): chunks(pdf_bytes((None,)), 'scan.pdf')
        with self.assertRaisesRegex(DocumentParseError, '第 2 页'): chunks(pdf_bytes(('Text', None)), 'mixed.pdf')
        writer = PdfWriter()
        writer.add_blank_page(500, 500)
        writer.encrypt('secret')
        buffer = BytesIO(); writer.write(buffer)
        with self.assertRaisesRegex(DocumentParseError, '加密'): chunks(buffer.getvalue(), 'secret.pdf')
        with self.assertRaisesRegex(DocumentParseError, '5 MiB'): chunks(b'x' * (MAX_SOURCE_BYTES + 1))

    @unittest.skipUnless(shutil.which('textutil'), 'macOS textutil unavailable')
    def test_legacy_doc_with_real_local_converter(self):
        result = subprocess.run(['textutil', '-convert', 'doc', '-format', 'txt', '-stdin', '-stdout', '-inputencoding', 'UTF-8'], input='Word 旧文档：桃园结义'.encode(), capture_output=True, check=True, timeout=20)
        self.assertIn('桃园结义', extract_sections(result.stdout, 'legacy.doc')[0].text)
        with patch('interview_rag.document_processing.legacy_word_converter', return_value=None):
            with self.assertRaisesRegex(DocumentParseError, '转换器'): extract_sections(result.stdout, 'legacy.doc')


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / '.venv', prefix='document-test-')
        self.root = Path(self.temp.name)
        self.notes = self.root / 'corpus/fixtures'
        self.notes.mkdir(parents=True)
        self.settings = Settings(_env_file=None, workspace_root=self.root, manifest_path=self.root / 'manifest.json', index_state_path=self.root / 'state.json', qdrant_local_path=self.root / 'qdrant', qdrant_mode='local')
        self.runtime = defaults()

    async def asyncTearDown(self):
        for key in list(_stores):
            if str(self.root) in key:
                _stores.pop(key)._client.close()
        self.temp.cleanup()

    def prepare(self):
        for name, raw in [('novel.txt', '三国演义\n桃园结义：刘备、关羽、张飞。'.encode()), ('book.PDF', pdf_bytes()), ('word.docx', docx_bytes())]:
            (self.notes / name).write_bytes(raw)
        manifest = Manifest(corpus_id='agent-interview')
        candidates = scan_candidates(manifest, self.root)
        self.assertEqual({c['format'] for c in candidates}, {'TXT', 'PDF', 'DOCX'})
        self.assertEqual(next(c for c in candidates if c['format'] == 'TXT')['title'], '三国演义')
        manifest = add_documents(manifest, candidates)
        save_manifest(self.settings.manifest_path, manifest)
        return manifest

    async def test_local_ingestion_reuse_update_and_pdf_error(self):
        manifest = self.prepare()
        ids = [d.doc_id for d in manifest.documents]
        async def embed(_config, inputs): return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]
        with patch('interview_rag.application.ingest.embed_texts', new=AsyncMock(side_effect=embed)) as mock:
            first = await ingest_documents(self.settings, self.runtime, ids)
            self.assertGreater(first.chunk_count, 0)
            mock.reset_mock()
            second = await ingest_documents(self.settings, self.runtime, ids)
            mock.assert_not_awaited()
            self.assertEqual(set(second.reused), set(ids))
            doc = next(d for d in manifest.documents if d.path.endswith('.txt'))
            (self.root / doc.path).write_text('三国演义\n青龙偃月刀重八十二斤。', encoding='utf-8')
            refresh_document(doc, self.root)
            save_manifest(self.settings.manifest_path, manifest)
            updated = await ingest_documents(self.settings, self.runtime, ids)
            self.assertEqual(updated.chunk_count, first.chunk_count)
            mock.assert_awaited()
            hits = store_for_settings(self.settings).search(updated.collection, [1.0, 0.0, 0.0, 0.0], 20)
            self.assertTrue(any('八十二斤' in h['text'] for h in hits))
            self.assertFalse(any('刘备、关羽' in h['text'] for h in hits))
            pdf = next(d for d in manifest.documents if d.path.endswith('.PDF'))
            (self.root / pdf.path).write_bytes(b'invalid PDF')
            refresh_document(pdf, self.root)
            save_manifest(self.settings.manifest_path, manifest)
            before = self.settings.index_state_path.read_bytes()
            mock.reset_mock()
            from interview_rag.application.ingest import IngestError
            with self.assertRaisesRegex(IngestError, 'PDF'):
                await ingest_documents(self.settings, self.runtime, ids)
            mock.assert_not_awaited()
            self.assertEqual(self.settings.index_state_path.read_bytes(), before)
            self.assertEqual(store_for_settings(self.settings).count(updated.collection), first.chunk_count)

    async def test_snapshot_preserves_binary_and_worker_reads_it(self):
        manifest = self.prepare()
        session = AsyncMock()
        session.add = unittest.mock.MagicMock()
        session.scalar.return_value = None
        manifest_hash, raw_map = await freeze_documents(session, manifest, manifest.documents, self.root)
        from interview_rag.storage.models import DocumentVersion
        versions = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], DocumentVersion)]
        self.assertEqual(len(versions), 3)
        self.assertTrue(any(v.raw.startswith(b'%PDF-') for v in versions))
        for doc in manifest.documents:
            self.assertEqual(raw_map[doc.doc_id], (self.root / doc.path).read_bytes())
        session.scalar.side_effect = [SimpleNamespace(manifest_json=manifest.model_dump_json())] + [next(v for v in versions if v.doc_id == doc.doc_id) for doc in manifest.documents]
        session.execute.return_value = SimpleNamespace(scalars=lambda:SimpleNamespace(all=lambda:[]))
        store = store_for_settings(self.settings)
        async def embed(_config, inputs): return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]
        with patch('interview_rag.application.build._store', return_value=store), patch('interview_rag.application.build.embed_texts', new=AsyncMock(side_effect=embed)):
            gid, count = await build_generation(session, self.settings, self.runtime, manifest_hash, [d.doc_id for d in manifest.documents], 'agent-interview')
        self.assertGreater(count, 0)
        self.assertEqual(store.count('gen_' + gid), count)


if __name__ == '__main__': unittest.main()
