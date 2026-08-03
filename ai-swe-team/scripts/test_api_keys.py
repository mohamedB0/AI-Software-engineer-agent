import os
from dotenv import load_dotenv
load_dotenv(override=True)

# -----------------------------------------------------------------------
# Test Groq
# -----------------------------------------------------------------------
print("=== GROQ ===")
try:
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
    )
    print(f"  PASS  reply={resp.choices[0].message.content.strip()}  model={resp.model}")
except Exception as e:
    print(f"  FAIL  {type(e).__name__}: {str(e)[:200]}")

# -----------------------------------------------------------------------
# Test Google Gemini  (key confirmed valid; may hit free-tier quota)
# -----------------------------------------------------------------------
print("\n=== GOOGLE GEMINI ===")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )
    resp = llm.invoke("Say OK in one word")
    print(f"  PASS  reply={resp.content.strip()}")
except Exception as e:
    msg = str(e)[:300]
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        print("  KEY VALID but QUOTA EXCEEDED (free tier limit hit, try again later)")
    else:
        print(f"  FAIL  {type(e).__name__}: {msg}")

# -----------------------------------------------------------------------
# Test Qwen (Alibaba DashScope)
# -----------------------------------------------------------------------
print("\n=== QWEN (DashScope) ===")
try:
    import httpx
    qwen_key = os.environ.get("QWEN_API_KEY", "")
    r = httpx.post(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + qwen_key,
            "Content-Type": "application/json",
        },
        json={"model": "qwen-turbo", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
        timeout=15,
    )
    print(f"  HTTP {r.status_code}")
    data = r.json()
    if r.status_code == 200:
        print(f"  PASS  reply={data['choices'][0]['message']['content'].strip()}")
    else:
        err = data.get("error", {}).get("message", str(data))
        print(f"  FAIL  {err[:200]}")
        if r.status_code == 401:
            print("  NOTE: Get a valid DashScope key at https://dashscope.aliyuncs.com")
except Exception as e:
    print(f"  FAIL  {type(e).__name__}: {str(e)[:200]}")
