#!/usr/bin/env python3
"""
精炼版：从 consensus_merged.json 第0篇文档（心脏外科学ch21）中提取临床知识
重点：阈值、药物策略、因果关系、可调控参数
减少噪音：对指标匹配加上下文约束
"""

import json
import re

with open("/home/user/SHU/consensus_merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

full_text = data[0]["text"]
total_len = len(full_text)

# ============================================================
# 辅助函数：按句子切分（考虑中文标点和换行）
# ============================================================
def split_sentences(text):
    """将文本切分为句子列表"""
    # 按中文句号、感叹号、问号、分号、换行切分
    parts = re.split(r'(?<=[。！？\n])', text)
    results = []
    for p in parts:
        p = p.strip()
        if len(p) >= 10:
            results.append(p)
    return results

# ============================================================
# 跟踪章节标题
# ============================================================
def extract_sections(text):
    """提取所有章节标题及其位置"""
    sections = []
    for m in re.finditer(r'(?:#{1,5}\s+|(?:第[一二三四五六七八九十百千]+[节章篇]\s*))(.*?)(?:\n|$)', text):
        sections.append((m.start(), m.group(0).strip().replace('#', '').strip()))
    return sections

sections = extract_sections(full_text)

def get_section_for_pos(pos):
    """根据字符位置获取所在章节"""
    result = "文档开头"
    for s_pos, s_name in sections:
        if s_pos <= pos:
            result = s_name
        else:
            break
    return result

# ============================================================
# 知识提取规则
# ============================================================

knowledge_entries = []
entry_id = 0

def add_entry(category, section, content, indicators=None, details=None):
    global entry_id
    entry_id += 1
    knowledge_entries.append({
        "id": entry_id,
        "category": category,
        "section": section,
        "content": content.strip(),
        "indicators": indicators or [],
        "details": details or "",
    })

# ============================================================
# 逐段扫描
# ============================================================
SEGMENT = 5000
for seg_start in range(0, total_len, SEGMENT):
    seg_end = min(seg_start + SEGMENT, total_len)
    segment = full_text[seg_start:seg_end]
    section = get_section_for_pos(seg_start)

    sentences = split_sentences(segment)

    for sent in sentences:
        # 更新章节
        sec_match = re.search(r'(?:#{1,5}\s+|第[一二三四五六七八九十百千]+[节章])', sent)
        if sec_match:
            header = re.search(r'(?:#{1,5}\s+|第[一二三四五六七八九十百千]+[节章]\s*)(.*?)$', sent)
            if header:
                section = header.group(0).strip().replace('#', '').strip()

        # ============ 规则1: 具体数值阈值 ============
        # 匹配含有明确数值+单位的句子
        threshold_patterns = [
            # 血流动力学阈值
            (r'(?:MAP|平均动脉压|收缩压|舒张压|血压).*?(\d+[\.\d]*)\s*(?:mmHg|mm\s*Hg|毫米汞柱)', ["MAP/血压"]),
            (r'(?:CVP|中心静脉压).*?(\d+)', ["CVP"]),
            (r'(?:PCWP|肺毛细血管楔压|肺楔压|肺动脉楔压).*?(\d+)', ["PCWP"]),
            (r'(?:肺血管阻力|PVR|Rp).*?(\d+[\.\d]*)\s*(?:Wood|wood|伍德)', ["PVR"]),
            (r'(?:跨肺压差|TPG).*?(\d+[\.\d]*)', ["跨肺压差"]),
            (r'(?:心率|HR|次/分|bpm).*?(\d+)', ["心率"]),
            (r'(?:心[指排]数|CI|心输出量|CO|心排血量).*?(\d+[\.\d]*)\s*(?:L|ml)', ["CI/CO"]),
            (r'(?:射血分数|EF|LVEF|RVEF).*?(\d+[\.\d]*)\s*%', ["EF"]),
            (r'射血分数[<＜>＞≤≥]?\s*(\d+)', ["EF"]),
            # 实验室检查阈值
            (r'(?:肌酐|Cr|creatinine).*?(\d+[\.\d]*)\s*(?:mg|mmol|mL)', ["肌酐"]),
            (r'肌酐清除率.*?(\d+[\.\d]*)\s*(?:mL|ml)', ["肌酐清除率"]),
            (r'肾小球滤过率.*?(\d+[\.\d]*)\s*(?:mL|ml)', ["GFR"]),
            (r'(?:胆红素|bilirubin).*?(\d+[\.\d]*)\s*(?:mg|μmol)', ["胆红素"]),
            (r'(?:乳酸|lactate).*?(\d+[\.\d]*)', ["乳酸"]),
            (r'(?:血红蛋白|Hb|HCT|红细胞压积|碳氧血红蛋白).*?(\d+[\.\d]*)\s*(?:%|g)', ["Hb/HCT"]),
            (r'(?:白细胞|WBC).*?(\d+)', ["WBC"]),
            (r'(?:血糖|glucose).*?(\d+[\.\d]*)', ["血糖"]),
            (r'FEV1.*?(\d+[\.\d]*)', ["FEV1"]),
            (r'FVC.*?(\d+[\.\d]*)', ["FVC"]),
            # 温度
            (r'(?:温度|温|℃|°C).*?(\d+[\.\d]*)\s*(?:℃|°C|度)', ["温度"]),
            (r'(\d+[\.\d]*)\s*(?:℃|°C|∘C)', ["温度"]),
            # 缺血时间
            (r'缺血时间.*?(\d+[\.\d]*)\s*(?:h|小时|min)', ["缺血时间"]),
            # PRA
            (r'(?:PRA|群体反应性抗体).*?(\d+[\.\d]*)\s*%', ["PRA"]),
            # 体外循环
            (r'(?:流量|灌注).*?(\d+[\.\d]*)\s*(?:L|ml|mL).*?(?:min|m)', ["灌注流量"]),
            (r'(?:VO2|最大氧耗|峰值摄氧量).*?(\d+[\.\d]*)', ["VO2"]),
        ]

        for pat, indicators in threshold_patterns:
            if re.search(pat, sent, re.IGNORECASE):
                add_entry("阈值", section, sent, indicators)
                break  # 一句话只分类一次到阈值

        # ============ 规则2: 药物策略（含剂量） ============
        drug_patterns = [
            # 免疫抑制剂
            (r'环孢素|环孢霉素|cyclosporine|CsA', ["环孢素"]),
            (r'他克莫司|tacrolimus|FK506|FK-506', ["他克莫司"]),
            (r'霉酚酸[酯脂]?|mycophenolate|MMF|骁悉', ["霉酚酸酯"]),
            (r'硫唑嘌呤|azathioprine|依木兰', ["硫唑嘌呤"]),
            (r'甲泼尼龙|methylprednisolone', ["甲泼尼龙"]),
            (r'泼尼松|prednisone', ["泼尼松"]),
            (r'糖皮质激素|皮质[类激]?[固醇素]', ["糖皮质激素"]),
            (r'抗胸腺细胞球蛋白|ATG|ATGAM', ["ATG"]),
            (r'巴利昔单抗|basiliximab', ["巴利昔单抗"]),
            (r'OKT3|莫罗单抗', ["OKT3"]),
            (r'利妥昔单抗|rituximab', ["利妥昔单抗"]),
            (r'西罗莫司|雷帕霉素|sirolimus|rapamycin|依维莫司|everolimus', ["mTOR抑制剂"]),
            # 血管活性药
            (r'去甲肾上腺素|norepinephrine|noradrenaline', ["去甲肾上腺素"]),
            (r'多巴胺|dopamine', ["多巴胺"]),
            (r'多巴酚丁胺|dobutamine', ["多巴酚丁胺"]),
            (r'(?<![异])肾上腺素(?!受体)|epinephrine|adrenaline', ["肾上腺素"]),
            (r'异丙肾上腺素|isoproterenol|isoprenaline', ["异丙肾上腺素"]),
            (r'米力农|milrinone', ["米力农"]),
            (r'左西孟旦|levosimendan', ["左西孟旦"]),
            (r'血管加压素|vasopressin', ["血管加压素"]),
            (r'硝普钠|nitroprusside', ["硝普钠"]),
            (r'硝酸甘油|nitroglycerin', ["硝酸甘油"]),
            (r'前列环素|前列腺素|prostacyclin|PGE1|前列地尔', ["前列腺素类"]),
            (r'一氧化氮|NO(?:\s|，|。)', ["一氧化氮"]),
            # 抗凝
            (r'肝素(?!化)', ["肝素"]),
            (r'华法林|warfarin', ["华法林"]),
            # 抗感染
            (r'更昔洛韦|ganciclovir', ["更昔洛韦"]),
            (r'复方磺胺甲噁唑|SMZ|TMP-SMX|甲氧苄啶', ["SMZ-TMP"]),
            (r'制霉菌素|nystatin', ["制霉菌素"]),
            (r'氟康唑|fluconazole', ["氟康唑"]),
            (r'阿昔洛韦|acyclovir', ["阿昔洛韦"]),
            # 机械支持
            (r'IABP|主动脉内球囊反搏', ["IABP"]),
            (r'ECMO|体外膜[肺氧]', ["ECMO"]),
            (r'(?:L|R)?VAD|心室辅助装置', ["VAD"]),
        ]

        for pat, drugs in drug_patterns:
            if re.search(pat, sent):
                # 判断是否含剂量信息
                has_dose = bool(re.search(r'\d+[\.\d]*\s*(?:mg|μg|g|ml|mL|U|IU|万)', sent))
                has_route = bool(re.search(r'静脉|口服|皮下|肌注|静注|静滴|泵入|灌注', sent))
                if has_dose or has_route or re.search(r'剂量|用[量法]|方案|每[日天]|bid|tid|qd|次/', sent):
                    add_entry("策略-药物剂量", section, sent, drugs, "含剂量/给药途径")
                elif re.search(r'毒性|副作用|不良|并发|禁忌|注意|监测', sent):
                    add_entry("策略-药物注意", section, sent, drugs, "药物注意事项")
                # 只记录有实际策略信息的
                break

        # ============ 规则3: 明确的因果关系 ============
        causal_patterns = [
            (r'(?:导致|引起|造成|引发|诱发).*(?:衰竭|损伤|障碍|死亡|出血|感染|排斥|并发|风险|休克)', "因果-不良后果"),
            (r'(?:导致|引起|造成).*(?:升高|降低|增加|减少|下降|增大)', "因果-参数变化"),
            (r'(?:升高|降低|增[加高大]|减[少低]).*(?:风险|死亡率|病死率|存活率|预后)', "因果-风险"),
            (r'(?:低|高|升高|降低)(?:MAP|血压|心率|心排|EF|射血分数|肺血管阻力|肌酐|胆红素|乳酸|氧).*(?:导致|引起|提示|预示|增加|死亡|风险|预后)', "因果-指标后果"),
            (r'危险因素|风险因素|预后因素|独立预测', "因果-危险因素"),
            (r'相对危险度|风险比|优势比|OR\s*[=＝]|HR\s*[=＝]|RR\s*[=＝]', "因果-统计关联"),
        ]

        for pat, causal_type in causal_patterns:
            if re.search(pat, sent):
                add_entry("因果关系", section, sent, details=causal_type)
                break

        # ============ 规则4: 体外循环/灌注策略 ============
        perfusion_patterns = [
            (r'(?:灌注|体外循环|CPB).*(?:温度|流量|时间|压力|管理)', "灌注管理"),
            (r'停搏液|心肌保护液|含血停搏|晶体停搏|del Nido|HTK|UW', "停搏液/保存液"),
            (r'(?:复温|降温|深低温|中低温|常温).*(?:灌注|循环|体外)', "温度管理"),
            (r'(?:PEEP|潮气量|通气|呼吸机).*(?:\d+|设置|参数|调节)', "呼吸管理"),
        ]

        for pat, strategy_type in perfusion_patterns:
            if re.search(pat, sent):
                add_entry("策略-灌注/管理", section, sent, details=strategy_type)
                break

        # ============ 规则5: 适应证/禁忌证 ============
        if re.search(r'适应[证症]|禁忌[证症]|绝对禁忌|相对禁忌|不宜|不应|不推荐|推荐|建议', sent):
            if len(sent) > 15:  # 避免纯标题
                add_entry("策略-适应证/禁忌证", section, sent)

        # ============ 规则6: 可调控性标注 ============
        controllable_patterns = [
            (r'(?:调[节整控]|维持|保持|控制).*(?:MAP|血压|心率|温度|体温|流量|CVP|心排|尿量|血糖|电解质)', "可直接调控"),
            (r'(?:目标|靶|维持).*(?:值|范围|水平).*\d+', "目标值可调"),
        ]

        for pat, ctrl_type in controllable_patterns:
            if re.search(pat, sent):
                add_entry("可调控性", section, sent, details=ctrl_type)
                break

# ============================================================
# 去重
# ============================================================
seen = set()
unique_entries = []
for e in knowledge_entries:
    key = e["content"][:100]
    if key not in seen:
        seen.add(key)
        unique_entries.append(e)

# ============================================================
# 输出
# ============================================================
print(f"{'='*80}")
print(f"  文档总长: {total_len} 字符 | 提取知识条目: {len(unique_entries)} 条 (去重前 {len(knowledge_entries)} 条)")
print(f"{'='*80}\n")

# 按类别统计
from collections import Counter
cat_counts = Counter(e["category"] for e in unique_entries)
print("--- 分类统计 ---")
for cat, cnt in cat_counts.most_common():
    print(f"  {cat}: {cnt} 条")
print()

# 按类别分组输出
categories_order = ["阈值", "策略-药物剂量", "策略-药物注意", "策略-灌注/管理", "策略-适应证/禁忌证", "因果关系", "可调控性"]

for cat in categories_order:
    items = [e for e in unique_entries if e["category"] == cat]
    if not items:
        continue
    print(f"\n{'='*80}")
    print(f"  【{cat}】 共 {len(items)} 条")
    print(f"{'='*80}")

    for i, item in enumerate(items, 1):
        text = item["content"]
        if len(text) > 250:
            text = text[:250] + "..."
        print(f"\n  [{i}] 章节: {item['section']}")
        if item["indicators"]:
            print(f"      指标: {', '.join(item['indicators'])}")
        if item["details"]:
            print(f"      类型: {item['details']}")
        print(f"      内容: {text}")

print(f"\n{'='*80}")
print("  提取完成")
print(f"{'='*80}")
