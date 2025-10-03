import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_deepseek_api():
    """Test DeepSeek API connection directly"""
    
    # Get API key
    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    print(f"API Key starts with: {api_key[:10]}..." if api_key else "No API key found")
    
    if not api_key:
        print("❌ No API key found in environment variables")
        return
    
    # API configuration
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple test payload
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and confirm the API is working."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print("\n🔄 Testing DeepSeek API connection...")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout error: {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_deepseek_api()