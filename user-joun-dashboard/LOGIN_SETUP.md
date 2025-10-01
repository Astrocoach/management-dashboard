# 🔐 Login Setup Instructions

## Setting Up Authentication

Your dashboard now includes secure login functionality that protects access to your analytics data.

### 1. Configure Your Credentials

Edit the `.env` file in your project root directory:

```bash
# Login Credentials - Keep this file secure and never commit to Git
ADMIN_EMAIL=your_email@example.com
ADMIN_PASSWORD=your_secure_password_here

# Optional: Session timeout in hours
SESSION_TIMEOUT_HOURS=24
```

### 2. Security Features

✅ **Environment Variables**: Credentials are stored in `.env` file, never in code  
✅ **Git Protection**: `.env` file is automatically excluded from Git commits  
✅ **Session Management**: Secure session handling with Streamlit  
✅ **Password Protection**: Password fields are masked during input  

### 3. Default Credentials

**⚠️ IMPORTANT: Change these immediately!**

- **Email**: `admin@example.com`
- **Password**: `your_secure_password_here`

### 4. How to Login

1. Navigate to your dashboard URL (usually `http://localhost:8501`)
2. You'll see a login page with email and password fields
3. Enter your credentials from the `.env` file
4. Click "Login" to access the dashboard
5. Use the "Logout" button in the sidebar when finished

### 5. Adding More Users

To add additional users, you can extend the `.env` file:

```bash
# Additional users
USER2_EMAIL=user2@example.com
USER2_PASSWORD=another_secure_password
```

Then modify the `check_credentials()` function in `main.py` to check multiple users.

### 6. Security Best Practices

- Use strong, unique passwords
- Never share your `.env` file
- Regularly update your credentials
- Use HTTPS in production environments
- Consider implementing password hashing for production use

### 7. Troubleshooting

**Login not working?**
- Check that your `.env` file exists and has the correct format
- Verify the email and password match exactly (case-sensitive)
- Ensure the `python-dotenv` package is installed

**Can't access dashboard?**
- Make sure you're using the correct URL
- Check that the Streamlit server is running
- Look for any error messages in the terminal

---

🔒 **Your dashboard is now secure!** Only users with valid credentials can access your analytics data.