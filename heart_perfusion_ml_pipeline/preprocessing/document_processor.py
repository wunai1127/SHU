"""
Document Processor - 文献数据预处理模块
处理PubMed文献和中国心脏移植指南文档
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class Document:
    """文档数据结构"""
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Optional[List[float]] = None


@dataclass
class Chunk:
    """文本块数据结构"""
    chunk_id: str
    doc_id: str
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    """文档处理器 - 处理心脏移植相关文献"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # 医学术语词典 - 用于实体识别
        self.medical_terms = self._load_medical_terms()

    def _load_medical_terms(self) -> Dict[str, str]:
        """加载医学术语词典"""
        return {
            # 生理指标
            "lactate": "生理指标",
            "乳酸": "生理指标",
            "pH": "生理指标",
            "pressure": "生理指标",
            "灌注压": "生理指标",
            "flow": "生理指标",
            "冠脉流量": "生理指标",
            "temperature": "生理指标",
            "温度": "生理指标",
            "缺血时间": "生理指标",
            "ischemic time": "生理指标",

            # 干预措施
            "加压": "干预措施",
            "减压": "干预措施",
            "灌注": "干预措施",
            "perfusion": "干预措施",
            "多巴胺": "药物",
            "dopamine": "药物",
            "肾上腺素": "药物",
            "epinephrine": "药物",

            # 并发症
            "PGD": "并发症",
            "primary graft dysfunction": "并发症",
            "原发性移植物功能障碍": "并发症",
            "rejection": "并发症",
            "排斥反应": "并发症",
            "心力衰竭": "并发症",
            "heart failure": "并发症",

            # 设备
            "OCS": "设备参数",
            "XVIVO": "设备参数",
            "EVLP": "设备参数",
            "NMP": "设备参数",
            "HMP": "设备参数",
        }

    def load_pubmed_data(self, json_path: str) -> List[Document]:
        """
        加载PubMed JSON数据

        Args:
            json_path: JSON文件路径

        Returns:
            Document列表
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for item in data:
            doc = Document(
                doc_id=item['id'],
                text=item['text'],
                metadata={
                    **item['metadata'],
                    'source_type': 'pubmed',
                    'processed_at': datetime.now().isoformat(),
                }
            )
            documents.append(doc)

        print(f"Loaded {len(documents)} PubMed documents")
        return documents

    def load_guideline_data(self, txt_dir: str) -> List[Document]:
        """
        加载中国心脏移植指南文档

        Args:
            txt_dir: 指南文件目录

        Returns:
            Document列表
        """
        documents = []
        txt_path = Path(txt_dir)

        for txt_file in txt_path.glob("*.txt"):
            if txt_file.name.startswith('.'):
                continue

            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # 解析文件名获取文档类型
            filename = txt_file.stem
            doc_type = "consensus" if "consensus" in filename else "book"

            doc = Document(
                doc_id=hashlib.md5(filename.encode()).hexdigest()[:12],
                text=text,
                metadata={
                    'source_file': str(txt_file),
                    'filename': filename,
                    'doc_type': doc_type,
                    'source_type': 'guideline',
                    'lang': 'zh',
                    'processed_at': datetime.now().isoformat(),
                }
            )
            documents.append(doc)

        print(f"Loaded {len(documents)} guideline documents")
        return documents

    def clean_text(self, text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊字符但保留医学符号
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\/\%\+\=\<\>\°\·\~μ]', '', text)

        # 标准化数字格式
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

        return text.strip()

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        将文档分块

        Args:
            doc: 文档对象

        Returns:
            Chunk列表
        """
        text = self.clean_text(doc.text)
        chunks = []

        # 按段落分割
        paragraphs = self._split_into_paragraphs(text)

        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + " "
            else:
                # 保存当前块
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = Chunk(
                        chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
                        doc_id=doc.doc_id,
                        text=current_chunk.strip(),
                        start_idx=current_start,
                        end_idx=current_start + len(current_chunk),
                        metadata={
                            **doc.metadata,
                            'chunk_index': len(chunks),
                        }
                    )
                    chunks.append(chunk)

                # 重叠处理
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_start = current_start + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + para + " "

        # 处理最后一块
        if len(current_chunk) >= self.min_chunk_size:
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
                doc_id=doc.doc_id,
                text=current_chunk.strip(),
                start_idx=current_start,
                end_idx=current_start + len(current_chunk),
                metadata={
                    **doc.metadata,
                    'chunk_index': len(chunks),
                }
            )
            chunks.append(chunk)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 使用多种分隔符
        paragraphs = re.split(r'\n\n|\n(?=[#\d])|(?<=[。！？])\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        提取医学实体（规则匹配 + 模式识别）

        Args:
            text: 输入文本

        Returns:
            实体列表
        """
        entities = []

        # 基于词典的实体识别
        for term, entity_type in self.medical_terms.items():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append({
                    'text': match.group(),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'method': 'dictionary'
                })

        # 数值+单位模式识别（如 "4.2 mmol/L", "75 mmHg"）
        numeric_patterns = [
            (r'(\d+\.?\d*)\s*(mmol/L|mmol/l)', '生理指标', 'lactate_value'),
            (r'(\d+\.?\d*)\s*(mmHg|mmhg)', '生理指标', 'pressure_value'),
            (r'(\d+\.?\d*)\s*(°C|℃)', '生理指标', 'temperature_value'),
            (r'(\d+\.?\d*)\s*(mL/min|ml/min|L/min)', '生理指标', 'flow_value'),
            (r'(\d+\.?\d*)\s*(%)', '生理指标', 'percentage_value'),
            (r'(\d+\.?\d*)\s*(h|hours?|小时)', '生理指标', 'time_value'),
        ]

        for pattern, entity_type, subtype in numeric_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'type': entity_type,
                    'subtype': subtype,
                    'value': float(match.group(1)),
                    'unit': match.group(2),
                    'start': match.start(),
                    'end': match.end(),
                    'method': 'pattern'
                })

        # 去重
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e['text'].lower(), e['start'], e['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return unique_entities

    def classify_document(self, doc: Document) -> str:
        """
        文档分类（用于后续检索优化）

        Args:
            doc: 文档对象

        Returns:
            文档类别
        """
        text_lower = doc.text.lower()

        # 关键词分类
        categories = {
            'donor_selection': ['donor selection', 'donor criteria', '供者', '供心选择'],
            'preservation': ['preservation', 'perfusion', 'EVLP', 'NMP', '保存', '灌注'],
            'transplant_surgery': ['transplant surgery', 'surgical technique', '移植手术', '手术技术'],
            'immunology': ['rejection', 'immunosuppression', 'HLA', '排斥', '免疫抑制'],
            'outcomes': ['survival', 'outcome', 'mortality', '生存', '预后'],
            'complications': ['complication', 'PGD', 'dysfunction', '并发症', '功能障碍'],
        }

        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)
        return 'general'

    def process_all(
        self,
        pubmed_path: str,
        guideline_dir: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        处理所有数据

        Args:
            pubmed_path: PubMed JSON路径
            guideline_dir: 指南目录
            output_dir: 输出目录

        Returns:
            处理统计信息
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_entities': 0,
            'categories': {},
        }

        all_chunks = []
        all_entities = []

        # 处理PubMed数据
        print("Processing PubMed documents...")
        pubmed_docs = self.load_pubmed_data(pubmed_path)

        # 处理指南数据
        print("Processing guideline documents...")
        guideline_docs = self.load_guideline_data(guideline_dir)

        all_docs = pubmed_docs + guideline_docs
        stats['total_documents'] = len(all_docs)

        # 处理每个文档
        for i, doc in enumerate(all_docs):
            if i % 1000 == 0:
                print(f"Processing document {i}/{len(all_docs)}")

            # 分块
            chunks = self.chunk_document(doc)
            all_chunks.extend([{
                'chunk_id': c.chunk_id,
                'doc_id': c.doc_id,
                'text': c.text,
                'start_idx': c.start_idx,
                'end_idx': c.end_idx,
                'metadata': c.metadata,
            } for c in chunks])

            # 实体提取
            entities = self.extract_entities(doc.text)
            for e in entities:
                e['doc_id'] = doc.doc_id
            all_entities.extend(entities)

            # 文档分类
            category = self.classify_document(doc)
            stats['categories'][category] = stats['categories'].get(category, 0) + 1

        stats['total_chunks'] = len(all_chunks)
        stats['total_entities'] = len(all_entities)

        # 保存处理结果
        with open(output_path / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        with open(output_path / 'entities.json', 'w', encoding='utf-8') as f:
            json.dump(all_entities, f, ensure_ascii=False, indent=2)

        with open(output_path / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\nProcessing complete!")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Categories: {stats['categories']}")

        return stats


if __name__ == "__main__":
    # 测试处理
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=128)

    # 示例用法
    # stats = processor.process_all(
    #     pubmed_path="data_extracted/heart_tx_all_merged_v8.json",
    #     guideline_dir="data_extracted/完整 txt",
    #     output_dir="outputs/processed_data"
    # )
