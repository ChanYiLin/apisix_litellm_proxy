#!/bin/bash
# chmod +x apisix/routes/setup-route.sh

ADMIN_KEY="${APISIX_ADMIN_KEY:-edd1c9f034335f136f87ad84b625c8f1}"
APISIX_ADMIN="${APISIX_ADMIN_URL:-http://localhost:9180}"
LITELLM_URL="${LITELLM_PROXY_URL:-http://litellm-proxy:8000}"

echo "Creating Bedrock chat completion route..."

curl -sf "${APISIX_ADMIN}/apisix/admin/routes/bedrock-chat-completion" \
  -X PUT \
  -H "X-API-KEY: ${ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"uri\": \"/v1/chat/completions\",
    \"methods\": [\"POST\"],
    \"plugins\": {
      \"ai-proxy-multi\": {
        \"fallback_strategy\": [\"rate_limiting\", \"http_5xx\"],
        \"balancer\": {
          \"algorithm\": \"roundrobin\"
        },
        \"logging\": {
          \"summaries\": true
        },
        \"instances\": [
          {
            \"name\": \"bedrock-us-east-1\",
            \"provider\": \"openai-compatible\",
            \"weight\": 5,
            \"priority\": 1,
            \"auth\": {
              \"header\": {
                \"X-LiteLLM-Instance\": \"bedrock-us-east-1\"
              }
            },
            \"options\": {
              \"model\": \"bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0\"
            },
            \"override\": {
              \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
            }
          },
          {
            \"name\": \"bedrock-ap-northeast-1\",
            \"provider\": \"openai-compatible\",
            \"weight\": 5,
            \"priority\": 0,
            \"auth\": {
              \"header\": {
                \"X-LiteLLM-Instance\": \"bedrock-ap-northeast-1\"
              }
            },
            \"options\": {
              \"model\": \"bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0\"
            },
            \"override\": {
              \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
            }
          }
        ]
      }
    }
  }" && echo "  [OK] bedrock-chat-completion route created" || echo "  [FAIL] bedrock-chat-completion route"

echo ""
echo "Creating health check route..."

curl -sf "${APISIX_ADMIN}/apisix/admin/routes/health" \
  -X PUT \
  -H "X-API-KEY: ${ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"uri\": \"/health\",
    \"methods\": [\"GET\"],
    \"plugins\": {
      \"proxy-rewrite\": {
        \"uri\": \"/health\"
      }
    },
    \"upstream\": {
      \"type\": \"roundrobin\",
      \"nodes\": {
        \"litellm-proxy:8000\": 1
      }
    }
  }" && echo "  [OK] health route created" || echo "  [FAIL] health route"
