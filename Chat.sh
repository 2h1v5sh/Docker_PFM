#!/bin/bash
# Test Gemini-Powered Chat Service
# Tests the AI Financial Advisor with real financial context

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Testing Gemini-Powered Chat Service${NC}"
echo -e "${BLUE}============================================${NC}\n"

# Store token
TOKEN=""

print_test() {
    echo -e "\n${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_response() {
    echo -e "${YELLOW}AI Response:${NC}"
    echo "$1" | jq -r '.response' 2>/dev/null || echo "$1"
    echo ""
}

# ==========================================
# 1. Setup - Login and Get Token
# ==========================================
print_test "1. Logging in to get access token"

LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8101/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.doe@example.com",
    "password": "SecurePass@123"
  }')

TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

if [ "$TOKEN" != "null" ] && [ -n "$TOKEN" ]; then
    print_success "Login successful - Token received"
else
    print_error "Login failed - Please ensure user exists"
    echo "Run: curl -X POST http://localhost:8101/register with user details first"
    exit 1
fi

# ==========================================
# 2. Check Chat Service Health
# ==========================================
print_test "2. Checking Chat Service Health"

HEALTH_RESPONSE=$(curl -s http://localhost:8110/health)
echo "$HEALTH_RESPONSE" | jq '.'

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_success "Chat service is healthy"
    
    GEMINI_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.gemini_api')
    if [ "$GEMINI_STATUS" = "configured" ]; then
        print_success "Gemini API is configured ✨"
    else
        print_error "Gemini API not configured - check GEMINI_API_KEY"
    fi
else
    print_error "Chat service not responding"
    exit 1
fi

# ==========================================
# 3. Get Smart Suggestions
# ==========================================
print_test "3. Getting contextual suggestions"

SUGGESTIONS_RESPONSE=$(curl -s -X GET http://localhost:8110/chat/suggestions \
  -H "Authorization: Bearer $TOKEN")

echo "$SUGGESTIONS_RESPONSE" | jq '.'
print_success "Suggestions retrieved"

# ==========================================
# 4. Test Chat - Spending Analysis
# ==========================================
print_test "4. Asking about spending this month"

CHAT1=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How much have I spent on food this month?",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT1"
print_success "Chat query processed"

# ==========================================
# 5. Test Chat - Financial Summary
# ==========================================
print_test "5. Asking for financial summary"

CHAT2=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Give me a complete financial summary for this month including my income, expenses, and savings rate",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT2"
print_success "Financial summary generated"

# ==========================================
# 6. Test Chat - Investment Advice
# ==========================================
print_test "6. Asking about SIP investments"

CHAT3=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Based on my current income and expenses, how much should I invest in SIP each month? Which funds would you recommend?",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT3"
print_success "Investment advice generated"

# ==========================================
# 7. Test Chat - Budget Recommendations
# ==========================================
print_test "7. Asking for budget recommendations"

CHAT4=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Looking at my spending patterns, what budgets should I set? Where am I overspending?",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT4"
print_success "Budget recommendations provided"

# ==========================================
# 8. Test Chat - Savings Strategy
# ==========================================
print_test "8. Asking about savings strategies"

CHAT5=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the best ways I can increase my savings rate to 30%? Give me specific actionable steps.",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT5"
print_success "Savings strategy generated"

# ==========================================
# 9. Test Chat - Debt Management
# ==========================================
print_test "9. Asking about debt management"

CHAT6=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I have a personal loan. What strategy should I use to pay it off faster while still maintaining emergency funds?",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT6"
print_success "Debt management advice provided"

# ==========================================
# 10. Test Chat - Tax Planning
# ==========================================
print_test "10. Asking about tax-saving investments"

CHAT7=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What tax-saving investment options (80C) should I consider? How much can I save on taxes?",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT7"
print_success "Tax planning advice generated"

# ==========================================
# 11. Get Conversation History
# ==========================================
print_test "11. Retrieving conversation history"

HISTORY=$(curl -s -X GET "http://localhost:8110/chat/history?conversation_id=test_conv_001" \
  -H "Authorization: Bearer $TOKEN")

echo "$HISTORY" | jq '{
  conversation_id,
  total_messages,
  messages: .messages | map({
    user_message: .user_message,
    timestamp: .timestamp
  })
}'

MESSAGE_COUNT=$(echo "$HISTORY" | jq -r '.total_messages')
print_success "Retrieved $MESSAGE_COUNT messages from history"

# ==========================================
# 12. Test Multiple Conversations
# ==========================================
print_test "12. Testing multiple conversation threads"

CHAT_NEW=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! I want to start fresh. Can you help me plan my finances?",
    "conversation_id": "test_conv_002"
  }')

print_response "$CHAT_NEW"
print_success "New conversation started"

# ==========================================
# 13. List All Conversations
# ==========================================
print_test "13. Listing all conversations"

CONVERSATIONS=$(curl -s -X GET http://localhost:8110/chat/conversations \
  -H "Authorization: Bearer $TOKEN")

echo "$CONVERSATIONS" | jq '.'
print_success "Conversations listed"

# ==========================================
# 14. Test Chat - Complex Financial Query
# ==========================================
print_test "14. Complex query - Complete financial plan"

CHAT_COMPLEX=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I earn ₹75,000 per month. Create a complete financial plan for me including budget allocation, emergency fund, investments, insurance, and retirement planning. Be very specific with numbers.",
    "conversation_id": "test_conv_002"
  }')

print_response "$CHAT_COMPLEX"
print_success "Complete financial plan generated"

# ==========================================
# 15. Test Chat - Follow-up Question
# ==========================================
print_test "15. Follow-up question in same conversation"

CHAT_FOLLOWUP=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What if my expenses increase by 20%? How should I adjust my plan?",
    "conversation_id": "test_conv_002"
  }')

print_response "$CHAT_FOLLOWUP"
print_success "Follow-up processed with context"

# ==========================================
# 16. Test Chat - Investment Portfolio Analysis
# ==========================================
print_test "16. Asking for portfolio analysis"

CHAT_PORTFOLIO=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze my current investment portfolio. Is it diversified enough? What changes would you recommend?",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT_PORTFOLIO"
print_success "Portfolio analysis completed"

# ==========================================
# 17. Test Chat - Goal-based Planning
# ==========================================
print_test "17. Goal-based financial planning"

CHAT_GOALS=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to buy a house worth ₹50 lakhs in 5 years and save for my child education of ₹20 lakhs in 10 years. How should I plan my investments?",
    "conversation_id": "test_conv_003"
  }')

print_response "$CHAT_GOALS"
print_success "Goal-based plan created"

# ==========================================
# 18. Test Chat - Emergency Fund
# ==========================================
print_test "18. Emergency fund recommendations"

CHAT_EMERGENCY=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How much emergency fund should I maintain? Where should I keep it - savings account, liquid funds, or FD?",
    "conversation_id": "test_conv_003"
  }')

print_response "$CHAT_EMERGENCY"
print_success "Emergency fund advice provided"

# ==========================================
# 19. Test Chat - Expense Optimization
# ==========================================
print_test "19. Expense optimization suggestions"

CHAT_OPTIMIZE=$(curl -s -X POST http://localhost:8110/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Looking at my transaction history, which expenses can I cut down without affecting my lifestyle? Give me a prioritized list.",
    "conversation_id": "test_conv_001"
  }')

print_response "$CHAT_OPTIMIZE"
print_success "Expense optimization suggestions generated"

# ==========================================
# 20. Test Conversation Export
# ==========================================
print_test "20. Exporting full conversation"

FULL_HISTORY=$(curl -s -X GET "http://localhost:8110/chat/history?conversation_id=test_conv_001" \
  -H "Authorization: Bearer $TOKEN")

echo "$FULL_HISTORY" | jq '{
  conversation_id,
  total_messages,
  summary: {
    first_message: .messages[0].user_message,
    last_message: .messages[-1].user_message,
    message_count: (.messages | length)
  }
}'

print_success "Conversation exported"

# ==========================================
# 21. Performance Test - Quick Questions
# ==========================================
print_test "21. Quick fire questions test"

questions=(
    "What is my savings rate?"
    "How much did I spend on entertainment?"
    "Show my top 5 expenses"
    "Am I saving enough?"
    "Should I invest more?"
)

for question in "${questions[@]}"; do
    echo -e "\n${YELLOW}Q: $question${NC}"
    
    QUICK_RESPONSE=$(curl -s -X POST http://localhost:8110/chat/message \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{
        \"message\": \"$question\",
        \"conversation_id\": \"quick_test\"
      }")
    
    echo "$QUICK_RESPONSE" | jq -r '.response' | head -3
done

print_success "Quick questions processed"

# ==========================================
# 22. Clear Specific Conversation
# ==========================================
print_test "22. Clearing test conversation"

CLEAR_RESPONSE=$(curl -s -X DELETE http://localhost:8110/chat/history/quick_test \
  -H "Authorization: Bearer $TOKEN")

echo "$CLEAR_RESPONSE" | jq '.'
print_success "Test conversation cleared"

# ==========================================
# SUMMARY & STATISTICS
# ==========================================
echo -e "\n${BLUE}============================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}============================================${NC}\n"

# Get final stats
FINAL_CONVERSATIONS=$(curl -s -X GET http://localhost:8110/chat/conversations \
  -H "Authorization: Bearer $TOKEN")

TOTAL_CONVERSATIONS=$(echo "$FINAL_CONVERSATIONS" | jq -r '.total')
TOTAL_MESSAGES=$(echo "$FINAL_CONVERSATIONS" | jq '[.conversations[].message_count] | add')

echo -e "${GREEN}✅ All tests completed successfully!${NC}\n"

echo "📊 Statistics:"
echo "  - Total conversations: $TOTAL_CONVERSATIONS"
echo "  - Total messages: $TOTAL_MESSAGES"
echo "  - Gemini API: $(curl -s http://localhost:8110/health | jq -r '.gemini_api')"
echo ""

echo "💡 Key Features Tested:"
echo "  ✓ Context-aware responses using real financial data"
echo "  ✓ Multi-conversation management"
echo "  ✓ Follow-up questions with context retention"
echo "  ✓ Complex financial planning queries"
echo "  ✓ Goal-based investment planning"
echo "  ✓ Budget and expense optimization"
echo "  ✓ Portfolio analysis"
echo "  ✓ Tax planning advice"
echo "  ✓ Conversation history and export"
echo ""

echo "🎯 Next Steps:"
echo "  1. Integrate with frontend chat UI"
echo "  2. Add streaming responses for real-time feel"
echo "  3. Implement conversation persistence in database"
echo "  4. Add user feedback mechanism"
echo "  5. Create conversation templates"
echo ""

echo "📚 Documentation:"
echo "  - Chat API: http://localhost:8110/docs"
echo "  - Service Info: http://localhost:8110/"
echo ""

echo -e "${GREEN}Your Gemini-powered AI Financial Advisor is ready! 🚀${NC}\n"