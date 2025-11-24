curl --location -g --request POST 'https://www.sophnet.com/api/open-apis/v1/chat/completions' \
--header "Authorization: Bearer $SOPHNET_API_KEY" \
--header "Content-Type: application/json" \
--data-raw '{
    "messages": [
        {
            "role": "system",
            "content": "你是SophNet的智能助手"
        },
        {
            "role": "user",
            "content": "你可以帮我做什么"
        }
    ],
    "model":"DeepSeek-V3.1"
}'