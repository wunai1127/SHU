"""
Agent基础框架
=============
- BaseAgent: Agent抽象基类，统一接口
- EventBus: 优先级消息总线（asyncio）
- AgentTool: 工具抽象，每个Agent持有若干Tool
- PatientState: Blackboard共享状态
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class Priority(Enum):
    CRITICAL = 0   # 立即处理
    HIGH = 1       # 高优先级
    NORMAL = 2     # 普通
    LOW = 3        # 低优先级（后台任务）


class AlertLevel(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    NORMAL = "normal"


class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"


# =============================================================================
# 核心数据结构
# =============================================================================

@dataclass
class AgentMessage:
    """Agent间通信消息"""
    source: str
    target: str          # 目标Agent名称, "*" = 广播
    msg_type: str        # "alert" | "query" | "result" | "command" | "data"
    payload: Any = None
    priority: int = 2    # 0=紧急 3=低
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""  # 关联ID，用于请求-响应匹配

    def __lt__(self, other):
        """优先级队列排序"""
        return self.priority < other.priority


@dataclass
class AlertEvent:
    """监测告警事件"""
    indicator: str
    value: float
    unit: str
    level: AlertLevel
    direction: str       # "high" | "low" | "abnormal"
    threshold: float
    deviation: float     # 偏离baseline程度
    trend: str           # "improving" | "stable" | "deteriorating"
    message: str = ""


@dataclass
class DiagnosisResult:
    """诊断分析结果"""
    primary_cause: str
    causal_chain: List[str]       # 因果链: A → B → C
    affected_indicators: List[str]
    indicator_type: str           # "setpoint" | "readout" | "injury_marker"
    upstream_setpoints: List[str] # 可调控的上游Setpoint
    confidence: float = 0.0


@dataclass
class StrategyRecommendation:
    """策略推荐"""
    indicator: str
    action: str
    drug: Optional[str] = None
    dose: Optional[str] = None
    target_value: Optional[float] = None
    target_range: Optional[Tuple[float, float]] = None
    reasoning_chain: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    severity: str = "warning"
    confidence: float = 0.0
    source: str = ""


@dataclass
class PatientState:
    """Blackboard - 全局共享患者状态"""
    sample_id: str = ""
    timestamp_min: int = 0
    measurements: Dict[str, float] = field(default_factory=dict)

    # 各Agent写入
    alerts: List[AlertEvent] = field(default_factory=list)
    diagnoses: List[DiagnosisResult] = field(default_factory=list)
    strategies: List[StrategyRecommendation] = field(default_factory=list)
    evidence_pool: List[Dict] = field(default_factory=list)

    # Coordinator写入
    risk_level: str = "UNKNOWN"
    final_summary: str = ""
    safety_warnings: List[str] = field(default_factory=list)
    drug_conflicts: List[str] = field(default_factory=list)

    # 元信息
    processing_log: List[str] = field(default_factory=list)


# =============================================================================
# Tool 抽象
# =============================================================================

@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: str = ""
    execution_time: float = 0.0


class AgentTool:
    """Agent工具封装 - 将现有后端模块包装为可调用工具"""

    def __init__(self, name: str, description: str, func: Callable, **kwargs):
        self.name = name
        self.description = description
        self._func = func
        self._kwargs = kwargs

    def execute(self, *args, **kwargs) -> ToolResult:
        """执行工具，统一错误处理和计时"""
        start = time.time()
        try:
            merged_kwargs = {**self._kwargs, **kwargs}
            result = self._func(*args, **merged_kwargs)
            return ToolResult(
                success=True,
                data=result,
                execution_time=time.time() - start
            )
        except Exception as e:
            logger.error(f"Tool [{self.name}] failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start
            )

    async def execute_async(self, *args, **kwargs) -> ToolResult:
        """异步执行（将同步函数放到线程池）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.execute(*args, **kwargs)
        )

    def __repr__(self):
        return f"Tool({self.name}: {self.description})"


# =============================================================================
# EventBus - 优先级消息总线
# =============================================================================

class EventBus:
    """
    Agent间通信总线
    - 优先级队列：critical消息优先处理
    - 支持定向发送和广播
    - 支持请求-响应模式（通过correlation_id）
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.PriorityQueue] = {}
        self._handlers: Dict[str, Callable] = {}
        self._response_waiters: Dict[str, asyncio.Future] = {}
        self._message_log: List[AgentMessage] = []
        self._running = False

    def register(self, agent_name: str, handler: Callable):
        """注册Agent及其消息处理函数"""
        self._queues[agent_name] = asyncio.PriorityQueue()
        self._handlers[agent_name] = handler
        logger.info(f"EventBus: registered agent [{agent_name}]")

    async def publish(self, msg: AgentMessage):
        """发布消息到目标Agent"""
        self._message_log.append(msg)

        if msg.target == "*":
            # 广播给所有Agent
            for name, queue in self._queues.items():
                if name != msg.source:
                    await queue.put((msg.priority, id(msg), msg))
        elif msg.target in self._queues:
            await self._queues[msg.target].put((msg.priority, id(msg), msg))

        # 检查是否有等待此响应的Future
        if msg.correlation_id and msg.correlation_id in self._response_waiters:
            future = self._response_waiters.pop(msg.correlation_id)
            if not future.done():
                future.set_result(msg)

    async def request(self, msg: AgentMessage, timeout: float = 10.0) -> Optional[AgentMessage]:
        """发送请求并等待响应（请求-响应模式）"""
        correlation_id = f"{msg.source}_{msg.target}_{time.time()}"
        msg.correlation_id = correlation_id

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._response_waiters[correlation_id] = future

        await self.publish(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._response_waiters.pop(correlation_id, None)
            logger.warning(f"EventBus: request timeout [{msg.source}→{msg.target}] ({timeout}s)")
            return None

    async def start(self):
        """启动消息分发循环"""
        self._running = True
        tasks = []
        for name in self._queues:
            tasks.append(asyncio.create_task(self._dispatch_loop(name)))
        await asyncio.gather(*tasks)

    async def _dispatch_loop(self, agent_name: str):
        """单Agent的消息分发循环"""
        queue = self._queues[agent_name]
        handler = self._handlers[agent_name]

        while self._running:
            try:
                _, _, msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                await handler(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"EventBus dispatch error [{agent_name}]: {e}")

    def stop(self):
        self._running = False

    def get_message_log(self) -> List[AgentMessage]:
        return self._message_log.copy()


# =============================================================================
# BaseAgent - Agent抽象基类
# =============================================================================

class BaseAgent(ABC):
    """
    Agent基类
    - 持有若干Tool
    - 通过EventBus通信
    - 读写PatientState(Blackboard)
    """

    def __init__(self, name: str, bus: EventBus, state: PatientState):
        self.name = name
        self.bus = bus
        self.state = state
        self.status = AgentStatus.IDLE
        self.tools: Dict[str, AgentTool] = {}

        # 注册到EventBus
        self.bus.register(self.name, self._handle_message)

    def register_tool(self, tool: AgentTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"Agent [{self.name}] registered tool: {tool.name}")

    def use_tool(self, tool_name: str, *args, **kwargs) -> ToolResult:
        """使用指定工具"""
        if tool_name not in self.tools:
            return ToolResult(success=False, error=f"Tool [{tool_name}] not found")
        return self.tools[tool_name].execute(*args, **kwargs)

    async def use_tool_async(self, tool_name: str, *args, **kwargs) -> ToolResult:
        """异步使用工具"""
        if tool_name not in self.tools:
            return ToolResult(success=False, error=f"Tool [{tool_name}] not found")
        return await self.tools[tool_name].execute_async(*args, **kwargs)

    async def send(self, target: str, msg_type: str, payload: Any = None,
                   priority: int = Priority.NORMAL.value):
        """发送消息给其他Agent"""
        msg = AgentMessage(
            source=self.name,
            target=target,
            msg_type=msg_type,
            payload=payload,
            priority=priority
        )
        await self.bus.publish(msg)

    async def request(self, target: str, msg_type: str, payload: Any = None,
                      timeout: float = 10.0) -> Optional[AgentMessage]:
        """请求-响应模式"""
        msg = AgentMessage(
            source=self.name,
            target=target,
            msg_type=msg_type,
            payload=payload,
            priority=Priority.HIGH.value
        )
        return await self.bus.request(msg, timeout=timeout)

    async def respond(self, original_msg: AgentMessage, payload: Any):
        """回复消息"""
        response = AgentMessage(
            source=self.name,
            target=original_msg.source,
            msg_type="result",
            payload=payload,
            priority=original_msg.priority,
            correlation_id=original_msg.correlation_id
        )
        await self.bus.publish(response)

    def log(self, message: str):
        """写入处理日志"""
        entry = f"[{self.name}] {message}"
        self.state.processing_log.append(entry)
        logger.info(entry)

    async def _handle_message(self, msg: AgentMessage):
        """消息分发入口"""
        self.status = AgentStatus.PROCESSING
        try:
            await self.on_message(msg)
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent [{self.name}] error handling message: {e}")
        finally:
            self.status = AgentStatus.IDLE

    @abstractmethod
    async def on_message(self, msg: AgentMessage):
        """子类实现：处理收到的消息"""
        pass

    @abstractmethod
    def setup_tools(self):
        """子类实现：初始化并注册工具"""
        pass

    def get_tool_list(self) -> List[Dict]:
        """获取工具列表（用于前端展示）"""
        return [
            {"name": t.name, "description": t.description}
            for t in self.tools.values()
        ]
