from typing import List, Optional, Dict, Any, Literal, Union
# 导入 Field
from pydantic import BaseModel, Extra, Field
# 导入 to_snake 用于生成别名
from pydantic.alias_generators import to_snake


from app.core.constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


# 辅助函数，将驼峰转为下划线
def _to_snake(name: str) -> str:
    # Pydantic v2 的 to_snake 可能不直接可用，或者行为略有不同
    # 手动实现一个简单的转换或使用 pydantic.alias_generators.to_snake
    # 这里我们直接使用 pydantic 提供的
    return to_snake(name)

class SafetySetting(BaseModel):
    category: Optional[
        Literal[
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]
    ] = None
    threshold: Optional[
        Literal[
            "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
            "BLOCK_LOW_AND_ABOVE",
            "BLOCK_MEDIUM_AND_ABOVE",
            "BLOCK_ONLY_HIGH",
            "BLOCK_NONE",
            "OFF",
        ]
    ] = None

    class Config:
        extra = Extra.allow
        # 允许通过字段名或别名填充
        populate_by_name = True
        # 为所有字段自动生成 snake_case 别名
        alias_generator = _to_snake


class GenerationConfig(BaseModel):
    # 字段名保持 camelCase，并添加 snake_case 别名
    stopSequences: Optional[List[str]] = Field(default=None, alias="stop_sequences")
    responseMimeType: Optional[str] = Field(default=None, alias="response_mime_type")
    responseSchema: Optional[Dict[str, Any]] = Field(default=None, alias="response_schema")
    candidateCount: Optional[int] = Field(default=1, alias="candidate_count")
    maxOutputTokens: Optional[int] = Field(default=None, alias="max_output_tokens")
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    topP: Optional[float] = Field(default=DEFAULT_TOP_P) # topP 已经是小写开头，但保持一致性
    topK: Optional[int] = Field(default=DEFAULT_TOP_K) # topK 已经是小写开头
    presencePenalty: Optional[float] = Field(default=None, alias="presence_penalty")
    frequencyPenalty: Optional[float] = Field(default=None, alias="frequency_penalty")
    responseLogprobs: Optional[bool] = Field(default=None, alias="response_logprobs")
    logprobs: Optional[int] = None
    thinkingConfig: Optional[Dict[str, Any]] = Field(default=None, alias="thinking_config")

    class Config:
        extra = Extra.allow
        # 允许通过字段名或别名填充
        populate_by_name = True


class SystemInstruction(BaseModel):
    role: str = "system"
    parts: List[Dict[str, Any]] | Dict[str, Any]

    class Config:
        extra = Extra.allow
        # 允许通过字段名或别名填充
        populate_by_name = True
        # 为所有字段自动生成 snake_case 别名 (如果需要的话)
        # alias_generator = _to_snake # role 和 parts 不需要别名


class GeminiContent(BaseModel):
    role: str
    parts: List[Dict[str, Any]]

    class Config:
        extra = Extra.allow
        # 允许通过字段名或别名填充
        populate_by_name = True
        # 为所有字段自动生成 snake_case 别名 (如果需要的话)
        # alias_generator = _to_snake # role 和 parts 不需要别名


class GeminiRequest(BaseModel):
    contents: List[GeminiContent] = []
    tools: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = []
    safetySettings: Optional[List[SafetySetting]] = Field(
        default=None, alias="safety_settings"
    )
    generationConfig: Optional[GenerationConfig] = Field(
        default=None, alias="generation_config"
    )
    systemInstruction: Optional[SystemInstruction] = Field(
        default=None, alias="system_instruction"
    )

    class Config:
        extra = Extra.allow
        populate_by_name = True


class ResetSelectedKeysRequest(BaseModel):
    keys: List[str]
    # 字段名保持 camelCase，并添加 snake_case 别名
    key_type: str = Field(alias="key_type") # key_type 本身就是 snake_case，但为了明确可以加

    class Config:
        # 允许通过字段名或别名填充
        populate_by_name = True


class VerifySelectedKeysRequest(BaseModel):
    keys: List[str]

    class Config:
        # 允许通过字段名或别名填充
        populate_by_name = True
        # 为所有字段自动生成 snake_case 别名 (如果需要的话)
        # alias_generator = _to_snake # keys 不需要别名
