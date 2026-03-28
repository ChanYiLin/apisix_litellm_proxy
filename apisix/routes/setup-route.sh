#!/bin/bash
# chmod +x apisix/routes/setup-route.sh

ADMIN_KEY="${APISIX_ADMIN_KEY:-edd1c9f034335f136f87ad84b625c8f1}"
APISIX_ADMIN="${APISIX_ADMIN_URL:-http://localhost:9180}"
LITELLM_URL="${LITELLM_PROXY_URL:-http://litellm-proxy:8000}"

# DEPRECATED: single bedrock-chat-completion route is replaced by per-model routes below.
# The old route had no vars matching and routed all /v1/chat/completions to Bedrock instances.
#
# curl -sf "${APISIX_ADMIN}/apisix/admin/routes/bedrock-chat-completion" \
#   -X PUT \
#   -H "X-API-KEY: ${ADMIN_KEY}" \
#   -H "Content-Type: application/json" \
#   -d "{
#     \"uri\": \"/v1/chat/completions\",
#     \"methods\": [\"POST\"],
#     \"plugins\": {
#       \"ai-proxy-multi\": {
#         \"fallback_strategy\": [\"rate_limiting\", \"http_5xx\"],
#         \"balancer\": {
#           \"algorithm\": \"roundrobin\"
#         },
#         \"logging\": {
#           \"summaries\": true
#         },
#         \"instances\": [
#           {
#             \"name\": \"bedrock-us-east-1\",
#             \"provider\": \"openai-compatible\",
#             \"weight\": 5,
#             \"priority\": 1,
#             \"auth\": {
#               \"header\": {
#                 \"X-LiteLLM-Instance\": \"bedrock-us-east-1\"
#               }
#             },
#             \"options\": {
#               \"model\": \"bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0\"
#             },
#             \"override\": {
#               \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
#             }
#           },
#           {
#             \"name\": \"bedrock-ap-northeast-1\",
#             \"provider\": \"openai-compatible\",
#             \"weight\": 5,
#             \"priority\": 0,
#             \"auth\": {
#               \"header\": {
#                 \"X-LiteLLM-Instance\": \"bedrock-ap-northeast-1\"
#               }
#             },
#             \"options\": {
#               \"model\": \"bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0\"
#             },
#             \"override\": {
#               \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
#             }
#           }
#         ]
#       }
#     }
#   }" && echo "  [OK] bedrock-chat-completion route created" || echo "  [FAIL] bedrock-chat-completion route"

# ---------------------------------------------------------------------------
# Per-model routes
# Each route matches on post_arg_model (the "model" field in JSON request body)
# and uses ai-proxy-multi to inject X-LiteLLM-Instance + handle failover.
# ---------------------------------------------------------------------------

echo "Creating bedrock claude-sonnet-4-5 route..."

curl -sf "${APISIX_ADMIN}/apisix/admin/routes/bedrock-claude-sonnet-4-5" \
  -X PUT \
  -H "X-API-KEY: ${ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"uri\": \"/v1/chat/completions\",
    \"methods\": [\"POST\"],
    \"vars\": [[\"post_arg_model\", \"==\", \"claude-sonnet-4-5\"]],
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
              \"model\": \"claude-sonnet-4-5\"
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
              \"model\": \"claude-sonnet-4-5\"
            },
            \"override\": {
              \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
            }
          }
        ]
      }
    }
  }" && echo "  [OK] bedrock-claude-sonnet-4-5 route created" || echo "  [FAIL] bedrock-claude-sonnet-4-5 route"

echo ""
echo "Creating gemini-2.0-flash route..."

curl -sf "${APISIX_ADMIN}/apisix/admin/routes/gemini-2-0-flash" \
  -X PUT \
  -H "X-API-KEY: ${ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"uri\": \"/v1/chat/completions\",
    \"methods\": [\"POST\"],
    \"vars\": [[\"post_arg_model\", \"==\", \"gemini-2.0-flash\"]],
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
            \"name\": \"gemini-flash-global\",
            \"provider\": \"openai-compatible\",
            \"weight\": 1,
            \"priority\": 1,
            \"auth\": {
              \"header\": {
                \"X-LiteLLM-Instance\": \"gemini-flash-global\"
              }
            },
            \"options\": {
              \"model\": \"gemini-2.0-flash\"
            },
            \"override\": {
              \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
            }
          }
        ]
      }
    }
  }" && echo "  [OK] gemini-2-0-flash route created" || echo "  [FAIL] gemini-2-0-flash route"

echo ""
echo "Creating gemini-2.5-pro route..."

curl -sf "${APISIX_ADMIN}/apisix/admin/routes/gemini-2-5-pro" \
  -X PUT \
  -H "X-API-KEY: ${ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"uri\": \"/v1/chat/completions\",
    \"methods\": [\"POST\"],
    \"vars\": [[\"post_arg_model\", \"==\", \"gemini-2.5-pro\"]],
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
            \"name\": \"gemini-pro-global\",
            \"provider\": \"openai-compatible\",
            \"weight\": 1,
            \"priority\": 1,
            \"auth\": {
              \"header\": {
                \"X-LiteLLM-Instance\": \"gemini-pro-global\"
              }
            },
            \"options\": {
              \"model\": \"gemini-2.5-pro\"
            },
            \"override\": {
              \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
            }
          }
        ]
      }
    }
  }" && echo "  [OK] gemini-2-5-pro route created" || echo "  [FAIL] gemini-2-5-pro route"

echo ""
echo "Creating vertex/gemini-2.0-flash route (multi-region failover)..."

curl -sf "${APISIX_ADMIN}/apisix/admin/routes/vertex-gemini-2-0-flash" \
  -X PUT \
  -H "X-API-KEY: ${ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"uri\": \"/v1/chat/completions\",
    \"methods\": [\"POST\"],
    \"vars\": [[\"post_arg_model\", \"==\", \"vertex/gemini-2.0-flash\"]],
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
            \"name\": \"vertex-flash-us\",
            \"provider\": \"openai-compatible\",
            \"weight\": 5,
            \"priority\": 1,
            \"auth\": {
              \"header\": {
                \"X-LiteLLM-Instance\": \"vertex-flash-us\"
              }
            },
            \"options\": {
              \"model\": \"vertex/gemini-2.0-flash\"
            },
            \"override\": {
              \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
            }
          },
          {
            \"name\": \"vertex-flash-asia\",
            \"provider\": \"openai-compatible\",
            \"weight\": 5,
            \"priority\": 0,
            \"auth\": {
              \"header\": {
                \"X-LiteLLM-Instance\": \"vertex-flash-asia\"
              }
            },
            \"options\": {
              \"model\": \"vertex/gemini-2.0-flash\"
            },
            \"override\": {
              \"endpoint\": \"${LITELLM_URL}/v1/chat/completions\"
            }
          }
        ]
      }
    }
  }" && echo "  [OK] vertex-gemini-2-0-flash route created" || echo "  [FAIL] vertex-gemini-2-0-flash route"

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
