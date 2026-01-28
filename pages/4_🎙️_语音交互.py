#!/usr/bin/env python3
"""
语音交互页面 - 实时灌注策略播报和语音问答
"""

import streamlit as st
import json
from pathlib import Path
import sys

# 添加src目录
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="语音交互", page_icon="🎙️", layout="wide")

st.title("🎙️ 语音交互助手")

# =============================================================================
# 语音功能 (使用浏览器 Web Speech API)
# =============================================================================

# 嵌入JavaScript实现语音功能
st.markdown("""
<style>
.voice-btn {
    padding: 15px 30px;
    font-size: 18px;
    border-radius: 25px;
    border: none;
    cursor: pointer;
    margin: 10px;
    transition: all 0.3s;
}
.voice-btn:hover {
    transform: scale(1.05);
}
.speak-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.listen-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}
.stop-btn {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
}
.status-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    font-size: 16px;
}
.listening {
    background: #fff3cd;
    border: 1px solid #ffc107;
}
.speaking {
    background: #d1ecf1;
    border: 1px solid #17a2b8;
}
.result-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    min-height: 100px;
}
.alert-critical {
    background: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 0 5px 5px 0;
}
.alert-warning {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 0 5px 5px 0;
}
</style>

<script>
// 语音合成 (TTS)
function speak(text, lang='zh-CN') {
    if ('speechSynthesis' in window) {
        // 停止当前播放
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang;
        utterance.rate = 0.9;  // 语速
        utterance.pitch = 1;   // 音调
        utterance.volume = 1;  // 音量

        // 尝试选择中文语音
        const voices = window.speechSynthesis.getVoices();
        const chineseVoice = voices.find(v => v.lang.includes('zh'));
        if (chineseVoice) {
            utterance.voice = chineseVoice;
        }

        utterance.onstart = () => {
            document.getElementById('status').innerHTML = '🔊 正在播报...';
            document.getElementById('status').className = 'status-box speaking';
        };
        utterance.onend = () => {
            document.getElementById('status').innerHTML = '✅ 播报完成';
            document.getElementById('status').className = 'status-box';
        };

        window.speechSynthesis.speak(utterance);
    } else {
        alert('您的浏览器不支持语音合成');
    }
}

// 停止播报
function stopSpeaking() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        document.getElementById('status').innerHTML = '⏹️ 已停止';
        document.getElementById('status').className = 'status-box';
    }
}

// 语音识别 (STT)
let recognition = null;

function startListening() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('您的浏览器不支持语音识别，请使用Chrome浏览器');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'zh-CN';
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart = () => {
        document.getElementById('status').innerHTML = '🎤 正在聆听...请说话';
        document.getElementById('status').className = 'status-box listening';
    };

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        document.getElementById('voice-input').value = transcript;

        // 如果是最终结果，发送到Streamlit
        if (event.results[event.results.length - 1].isFinal) {
            document.getElementById('status').innerHTML = '✅ 识别完成: ' + transcript;
            document.getElementById('status').className = 'status-box';

            // 触发Streamlit更新
            const inputEvent = new Event('input', { bubbles: true });
            document.getElementById('voice-input').dispatchEvent(inputEvent);
        }
    };

    recognition.onerror = (event) => {
        document.getElementById('status').innerHTML = '❌ 识别错误: ' + event.error;
        document.getElementById('status').className = 'status-box';
    };

    recognition.onend = () => {
        if (document.getElementById('status').innerHTML.includes('聆听')) {
            document.getElementById('status').innerHTML = '⏹️ 聆听结束';
            document.getElementById('status').className = 'status-box';
        }
    };

    recognition.start();
}

function stopListening() {
    if (recognition) {
        recognition.stop();
        document.getElementById('status').innerHTML = '⏹️ 已停止聆听';
        document.getElementById('status').className = 'status-box';
    }
}

// 页面加载时初始化语音列表
window.speechSynthesis.onvoiceschanged = () => {
    window.speechSynthesis.getVoices();
};
</script>

<div id="status" class="status-box">🎙️ 语音助手就绪</div>
""", unsafe_allow_html=True)

# =============================================================================
# 策略播报区域
# =============================================================================
st.markdown("---")
st.markdown("## 📢 实时策略播报")

col1, col2 = st.columns([2, 1])

with col1:
    # 模拟当前警报数据
    current_alerts = [
        {"level": "critical", "indicator": "MAP", "value": 45, "unit": "mmHg", "target": "65-80",
         "message": "平均动脉压严重偏低，建议立即使用去甲肾上腺素0.05到0.1微克每公斤每分钟"},
        {"level": "critical", "indicator": "K+", "value": 6.2, "unit": "mmol/L", "target": "3.5-5.0",
         "message": "血钾严重升高，存在心律失常风险，建议胰岛素加葡萄糖降钾治疗"},
        {"level": "warning", "indicator": "Lactate", "value": 4.5, "unit": "mmol/L", "target": "<4.0",
         "message": "乳酸轻度升高，提示组织灌注不足，需优化血流动力学"}
    ]

    st.markdown("### 当前警报")

    for alert in current_alerts:
        level_class = "alert-critical" if alert["level"] == "critical" else "alert-warning"
        level_icon = "🔴" if alert["level"] == "critical" else "🟡"

        st.markdown(f"""
        <div class="{level_class}">
            <strong>{level_icon} {alert['indicator']}: {alert['value']} {alert['unit']}</strong> (目标: {alert['target']})<br/>
            {alert['message']}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 播报控制")

    # 生成播报文本
    broadcast_text = "灌注监测警报播报。"
    for alert in current_alerts:
        level_text = "危急警报" if alert["level"] == "critical" else "警告"
        broadcast_text += f"{level_text}：{alert['indicator']}当前值{alert['value']}{alert['unit']}，{alert['message']}。"

    # 播报按钮
    st.markdown(f"""
    <button class="voice-btn speak-btn" onclick="speak(`{broadcast_text}`)">
        🔊 播报全部警报
    </button>
    <button class="voice-btn stop-btn" onclick="stopSpeaking()">
        ⏹️ 停止播报
    </button>
    """, unsafe_allow_html=True)

    # 单独播报选项
    st.markdown("#### 单独播报")
    for i, alert in enumerate(current_alerts):
        single_text = f"{alert['indicator']}当前值{alert['value']}{alert['unit']}，{alert['message']}"
        st.markdown(f"""
        <button class="voice-btn speak-btn" style="padding: 8px 15px; font-size: 14px;"
                onclick="speak(`{single_text}`)">
            🔊 {alert['indicator']}
        </button>
        """, unsafe_allow_html=True)

# =============================================================================
# 语音问答区域
# =============================================================================
st.markdown("---")
st.markdown("## 🎤 语音问答")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 语音输入")

    st.markdown("""
    <button class="voice-btn listen-btn" onclick="startListening()">
        🎤 开始语音输入
    </button>
    <button class="voice-btn stop-btn" onclick="stopListening()">
        ⏹️ 停止
    </button>
    <br/><br/>
    <input type="text" id="voice-input" style="width: 100%; padding: 10px; font-size: 16px; border-radius: 5px; border: 1px solid #ddd;"
           placeholder="语音识别结果将显示在这里..." />
    """, unsafe_allow_html=True)

    # 文字输入备选
    user_question = st.text_input("或直接输入问题:", key="text_question",
                                   placeholder="例如：MAP低应该怎么处理？")

with col2:
    st.markdown("### AI回答")

    # 预设问答库
    qa_database = {
        "MAP": {
            "keywords": ["MAP", "血压", "动脉压", "低血压"],
            "answer": "MAP偏低时，首先检查容量状态，若容量充足，建议使用去甲肾上腺素0.05到0.1微克每公斤每分钟，目标MAP大于65毫米汞柱。需每5分钟监测MAP和心率。"
        },
        "Lactate": {
            "keywords": ["乳酸", "Lactate", "lactate"],
            "answer": "乳酸升高提示组织灌注不足或缺氧。处理方法：优化血流动力学，改善组织氧供，必要时纠正贫血使血红蛋白大于10克每分升。每30分钟复查乳酸。"
        },
        "K": {
            "keywords": ["钾", "K+", "高钾", "低钾", "血钾"],
            "answer": "高钾血症处理：首先静脉推注10%葡萄糖酸钙10毫升稳定心肌膜，然后使用胰岛素10单位加50%葡萄糖50毫升促进钾内移。严重时考虑血液透析。每30分钟复查血钾和心电图。"
        },
        "CI": {
            "keywords": ["心指数", "CI", "心输出量"],
            "answer": "心指数偏低时，首选多巴酚丁胺5到10微克每公斤每分钟增强心肌收缩力。若效果不佳，可加用米力农。需持续监测CI、CVP和PCWP。"
        },
        "pH": {
            "keywords": ["pH", "酸中毒", "碱中毒", "酸碱"],
            "answer": "代谢性酸中毒时，首先查找原因如乳酸堆积、肾功能不全。轻度可通过改善灌注自行纠正，严重时可补充碳酸氢钠，根据碱剩余计算剂量。每30分钟复查血气。"
        }
    }

    # 处理问题
    if user_question:
        answer = "抱歉，我暂时无法回答这个问题。请咨询值班医生。"

        for topic, data in qa_database.items():
            for keyword in data["keywords"]:
                if keyword.lower() in user_question.lower():
                    answer = data["answer"]
                    break

        st.markdown(f"""
        <div class="result-box">
            <strong>问题：</strong>{user_question}<br/><br/>
            <strong>回答：</strong>{answer}
        </div>
        <button class="voice-btn speak-btn" onclick="speak(`{answer}`)">
            🔊 播报回答
        </button>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-box">
            <em>请输入或说出您的问题...</em><br/><br/>
            <strong>示例问题：</strong><br/>
            • MAP低应该怎么处理？<br/>
            • 乳酸升高怎么办？<br/>
            • 高钾血症如何处理？<br/>
            • 心指数偏低用什么药？
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# 自动播报设置
# =============================================================================
st.markdown("---")
st.markdown("## ⚙️ 自动播报设置")

col1, col2, col3 = st.columns(3)

with col1:
    auto_broadcast = st.checkbox("启用自动播报", value=False)

with col2:
    broadcast_interval = st.selectbox("播报间隔", ["每5分钟", "每10分钟", "每30分钟", "仅危急时"])

with col3:
    broadcast_level = st.multiselect("播报级别", ["危急 (Critical)", "警告 (Warning)"],
                                      default=["危急 (Critical)"])

if auto_broadcast:
    st.info("🔔 自动播报已启用。当检测到选定级别的异常时，系统将自动语音播报。")

# =============================================================================
# 使用说明
# =============================================================================
st.markdown("---")
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 语音播报功能
    - 点击 **🔊 播报全部警报** 播报当前所有警报
    - 点击单个指标按钮播报特定警报
    - 点击 **⏹️ 停止播报** 可随时停止

    ### 语音问答功能
    - 点击 **🎤 开始语音输入** 后对麦克风说话
    - 识别完成后系统会自动显示回答
    - 也可以直接在输入框输入文字问题
    - 点击 **🔊 播报回答** 听取语音回答

    ### 浏览器要求
    - 推荐使用 **Chrome** 浏览器以获得最佳语音体验
    - 首次使用需要允许麦克风权限
    - Safari/Firefox 可能不支持语音识别功能

    ### 支持的问题类型
    - 指标异常处理（MAP、乳酸、血钾、心指数、pH等）
    - 药物剂量建议
    - 监测频率建议
    """)
