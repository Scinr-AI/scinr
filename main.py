import asyncio
import os

from langchain_aws import ChatBedrockConverse

from scinr.newton import configure
from scinr.newton.pipeline import run_pipeline

_MODEL_ID = os.getenv("MODEL_ID")
_MAX_TOKENS = 65536


async def main():
    configure(
        ChatBedrockConverse(
            model=_MODEL_ID,
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            max_tokens=_MAX_TOKENS,
            temperature=0,
        ),
        enabled_base_themes=["default"],
        extra_models_paths=["./own_models"]
    )
    await run_pipeline(input_raw="./files/new-files")


if __name__ == "__main__":
    asyncio.run(main())
