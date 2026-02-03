from pydantic import BaseModel, Field


class TaskDescription(BaseModel):
    function_name: str
    description: str
    security_policy: str
    context: str
    arguments: str
    return_: str = Field(alias="return")
    raise_: str = Field(alias="raise")


class GroundTruth(BaseModel):
    code_before: str
    vulnerable_code: str
    patched_code: str
    code_after: str


class Unittest(BaseModel):
    setup: str
    testcases: str


class CWEData(BaseModel):
    CWE_ID: str
    CVE_ID: str = ""
    task_description: TaskDescription
    ground_truth: GroundTruth
    unittest: Unittest
    install_requires: list[str]
    rule: str = ""


class TestCodeParams(BaseModel):
    setup: str
    code: str
    testcases: str
    func_name: str
    install_requires: list[str]


CWE_use_rule = {
    295,
    367,
    732,
    400,
    338,
    611,
    22,
    78,
    120,
    281,
}
