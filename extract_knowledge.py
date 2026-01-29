#!/usr/bin/env python3
"""
从 consensus_merged.json 第0篇文档（心脏外科学ch21）中提取临床知识
按5000字符分段，搜索：特征/指标、阈值、策略、因果关系、可调控性
"""

import json
import re
import sys

# ============================================================
# 1. 加载文本
# ============================================================
with open("/home/user/SHU/consensus_merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

full_text = data[0]["text"]
total_len = len(full_text)
print(f"=== 文档总长度: {total_len} 字符 ===\n")

# ============================================================
# 2. 定义关键词/正则模式
# ============================================================

# 血流动力学 & 实验室指标关键词
INDICATOR_KW = [
    "MAP", "平均动脉压", "CI", "心指数", "心脏指数",
    "CVP", "中心静脉压", "PCWP", "肺毛细血管楔压", "肺动脉楔压",
    "心率", "HR", "乳酸", "肌钙蛋白", "cTnI", "cTnT", "BNP", "NT-proBNP",
    "pH", "PaO2", "PaCO2", "SaO2", "SpO2", "SvO2", "ScvO2",
    "LAP", "左房压", "左心房压", "RAP", "右房压", "右心房压",
    "EF", "射血分数", "LVEF", "RVEF",
    "温度", "体温", "核心温度", "鼻咽温", "膀胱温", "直肠温",
    "血红蛋白", "Hb", "HCT", "红细胞压积", "血小板",
    "肌酐", "尿量", "GFR", "eGFR", "肾小球滤过率",
    "胆红素", "ALT", "AST", "白蛋白",
    "INR", "PT", "APTT", "ACT", "凝血",
    "血糖", "血钾", "血钠", "血钙", "电解质",
    "心排血量", "心输出量", "CO", "SVR", "体循环阻力", "全身血管阻力",
    "PVR", "肺血管阻力", "肺动脉压", "PAP",
    "IABP", "ECMO", "VAD", "LVAD",
    "VO2", "氧耗", "氧供", "DO2", "氧输送",
    "CPB", "体外循环", "缺血时间", "冷缺血",
    "PRA", "群体反应性抗体",
    "FEV1", "FVC", "肺功能",
    "BMI", "体重指数",
    "CRRT", "透析",
    "白细胞", "WBC", "CRP", "降钙素原", "PCT",
]

# 数值阈值模式: 捕获 "xxx > 30", "xxx ≥ 2.5", "xxx为60~80", "xxx 10-20 mmHg" 等
THRESHOLD_PATTERNS = [
    r'[>＞≥≤<＜]\s*\d+[\.\d]*',
    r'\d+[\.\d]*\s*[~\-～至]\s*\d+[\.\d]*',
    r'\d+[\.\d]*\s*(?:mmHg|mmol|mg|μg|ng|ml|mL|L|g|kg|cm|mm|%|℃|°C|bpm|次/分|U/L|IU|ml/min|L/min|Woods)',
    r'目标[值为是：:]\s*\d+',
    r'维持[在于]\s*\d+',
    r'控制[在于]\s*\d+',
    r'不[应该宜]超过\d+',
    r'低于\s*\d+',
    r'高于\s*\d+',
    r'大于\s*\d+',
    r'小于\s*\d+',
    r'正常[值为范围]\s*[:：]?\s*\d+',
]

# 策略关键词
STRATEGY_KW = [
    "剂量", "用量", "mg/kg", "μg/kg", "mg/d", "μg/min",
    "静脉", "口服", "皮下", "肌注", "静注", "静滴", "泵入", "持续泵入",
    "灌注", "灌注液", "停搏液", "心肌保护", "心脏保护",
    "升压", "去甲肾上腺素", "多巴胺", "多巴酚丁胺", "肾上腺素",
    "米力农", "左西孟旦", "异丙肾上腺素", "血管加压素",
    "硝普钠", "硝酸甘油", "前列环素", "前列腺素",
    "免疫抑制", "环孢素", "他克莫司", "霉酚酸", "硫唑嘌呤",
    "甲泼尼龙", "泼尼松", "糖皮质激素", "抗胸腺细胞球蛋白", "ATG",
    "巴利昔单抗", "利妥昔单抗", "OKT3",
    "排斥反应", "抗排斥",
    "抗凝", "肝素", "华法林", "阿司匹林", "氯吡格雷",
    "利尿", "呋塞米", "甘露醇",
    "抗感染", "抗生素", "预防感染", "抗真菌", "抗病毒", "更昔洛韦",
    "复方磺胺甲噁唑", "SMZ",
    "制霉菌素", "氟康唑", "伊曲康唑",
    "低温", "深低温", "常温",
    "正性肌力", "血管扩张", "血管收缩",
    "输血", "红细胞悬液", "血浆", "血小板输注",
    "机械辅助", "机械循环",
    "补液", "液体管理", "容量管理",
    "通气", "呼吸机", "PEEP", "潮气量",
    "适应证", "禁忌证", "禁忌症",
    "方案", "方法", "治疗", "处理", "处置", "干预", "措施",
]

# 因果关系关键词
CAUSAL_KW = [
    "导致", "引起", "造成", "引发", "诱发",
    "由于", "因为", "因此", "所以", "故",
    "若", "如果", "则", "当.*时",
    "可能.*出现", "容易.*发生",
    "增加.*风险", "降低.*风险",
    "升高", "降低", "增加", "减少", "改善", "恶化",
    "提示", "表明", "说明", "预示", "反映",
    "与.*相关", "与.*有关",
    "并发症", "不良反应", "副作用",
    "危险因素", "风险因素", "预后因素",
]

# ============================================================
# 3. 分段处理
# ============================================================
SEGMENT_SIZE = 5000
knowledge_items = []
current_section = "文档开头"

def detect_section(text_segment):
    """检测当前段落所属章节标题"""
    # 查找最后一个出现的章节标题
    headers = list(re.finditer(r'(?:#{1,4}\s+|第[一二三四五六七八九十百]+[节章]\s*)(.*?)(?:\n|$)', text_segment))
    if headers:
        return headers[-1].group(0).strip().replace('#', '').strip()
    return None

def extract_sentences_with_context(text, start_pos):
    """按句子分割文本"""
    sentences = re.split(r'(?<=[。！？；\n])', text)
    results = []
    pos = 0
    for sent in sentences:
        if sent.strip():
            results.append((sent.strip(), start_pos + pos))
        pos += len(sent)
    return results

def classify_sentence(sentence):
    """对句子进行知识分类"""
    categories = []

    # 检查是否包含指标
    has_indicator = False
    matched_indicators = []
    for kw in INDICATOR_KW:
        if kw in sentence:
            has_indicator = True
            matched_indicators.append(kw)

    # 检查是否包含阈值数值
    has_threshold = False
    for pat in THRESHOLD_PATTERNS:
        if re.search(pat, sentence):
            has_threshold = True
            break
    # 额外检查：包含指标+数字组合
    if has_indicator and re.search(r'\d+', sentence):
        has_threshold = True

    # 检查是否包含策略
    has_strategy = False
    matched_strategies = []
    for kw in STRATEGY_KW:
        if kw in sentence:
            has_strategy = True
            matched_strategies.append(kw)

    # 检查是否包含因果关系
    has_causal = False
    for kw in CAUSAL_KW:
        if re.search(kw, sentence):
            has_causal = True
            break

    if has_threshold and has_indicator:
        categories.append("阈值")
    if has_indicator and not has_threshold:
        categories.append("特征/指标")
    if has_strategy:
        categories.append("策略")
    if has_causal:
        categories.append("因果关系")

    return categories, matched_indicators, matched_strategies

# ============================================================
# 4. 主循环：分段扫描
# ============================================================
segment_count = 0
for start in range(0, total_len, SEGMENT_SIZE):
    end = min(start + SEGMENT_SIZE, total_len)
    segment = full_text[start:end]
    segment_count += 1

    # 检测章节
    sec = detect_section(segment)
    if sec:
        current_section = sec

    # 提取句子
    sentences = extract_sentences_with_context(segment, start)

    for sent_text, sent_pos in sentences:
        if len(sent_text) < 8:  # 跳过太短的
            continue

        cats, indicators, strategies = classify_sentence(sent_text)

        if cats:  # 有匹配
            knowledge_items.append({
                "text": sent_text,
                "section": current_section,
                "categories": cats,
                "indicators": indicators,
                "strategies": strategies,
                "position": sent_pos,
            })

print(f"=== 共处理 {segment_count} 个段落, 提取 {len(knowledge_items)} 条知识 ===\n")

# ============================================================
# 5. 去重 & 过滤（去掉纯历史叙述、人名等噪音）
# ============================================================
NOISE_PATTERNS = [
    r'^[A-Z][a-z]+和[A-Z]',   # 纯人名开头
    r'^\d{4}年.*报道',         # 纯历史报道
    r'^在\d{4}年',
]

filtered = []
seen_texts = set()
for item in knowledge_items:
    text = item["text"]
    # 去重
    short_key = text[:80]
    if short_key in seen_texts:
        continue
    seen_texts.add(short_key)

    # 去噪
    is_noise = False
    for np in NOISE_PATTERNS:
        if re.match(np, text):
            # 但如果包含阈值或策略，保留
            if "阈值" not in item["categories"] and "策略" not in item["categories"]:
                is_noise = True
                break
    if is_noise:
        continue

    filtered.append(item)

print(f"=== 去重去噪后: {len(filtered)} 条知识 ===\n")

# ============================================================
# 6. 分类输出
# ============================================================
# 按类型分组
by_type = {"阈值": [], "特征/指标": [], "策略": [], "因果关系": []}
for item in filtered:
    for cat in item["categories"]:
        by_type[cat].append(item)

for cat_name in ["阈值", "策略", "因果关系", "特征/指标"]:
    items = by_type[cat_name]
    print(f"\n{'='*80}")
    print(f"  【{cat_name}】 共 {len(items)} 条")
    print(f"{'='*80}")
    for i, item in enumerate(items, 1):
        text = item["text"]
        # 截断过长的条目但保留完整信息
        if len(text) > 200:
            text = text[:200] + "..."
        section = item["section"]
        indicators = ", ".join(item["indicators"][:5]) if item["indicators"] else "-"
        strategies = ", ".join(item["strategies"][:5]) if item["strategies"] else "-"
        type_tags = " | ".join(item["categories"])
        print(f"\n[{i}] [{type_tags}] 章节: {section}")
        if item["indicators"]:
            print(f"    指标: {indicators}")
        if item["strategies"]:
            print(f"    策略: {strategies}")
        print(f"    内容: {text}")
