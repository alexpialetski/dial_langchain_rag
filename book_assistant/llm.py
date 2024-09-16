from langchain_aws import BedrockEmbeddings, ChatBedrock


embeddings_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
)

text_model = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs=dict(temperature=0),
)
