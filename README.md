# MCP Document Chatbot

A full-stack chatbot application that processes documents and provides intelligent responses using Model Context Protocol (MCP), LangChain, and modern web technologies.

## Features

- 📄 Document upload and processing (PDF, TXT, DOCX)
- 🤖 Intelligent chatbot with document-aware responses
- 🔍 Context-aware Q&A based on uploaded documents
- 👤 User authentication and session management
- 📊 Real-time chat interface
- 🗄️ Persistent storage with Supabase

## Tech Stack

### Frontend

- **React.js** - Modern UI framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client for API calls

### Backend

- **FastAPI** - High-performance Python web framework
- **LangChain** - LLM application framework
- **MCP (Model Context Protocol)** - Chatbot integration
- **Uvicorn** - ASGI server

### Database

- **Supabase** - PostgreSQL database with real-time features
- **Vector storage** - For document embeddings

### AI/ML

- **OpenAI API** - Language model integration
- **Document embeddings** - For semantic search
- **Vector similarity search** - Document retrieval

## Project Structure

```
ai-chatbot/
├── frontend/                 # React.js application
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/          # Application pages
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API service calls
│   │   └── utils/          # Utility functions
│   ├── public/
│   └── package.json
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Core functionality
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utility functions
│   ├── requirements.txt
│   └── main.py
├── docs/                    # Documentation
└── README.md
```

## Prerequisites

- Node.js 18+
- Python 3.9+
- Supabase account
- OpenAI API key

## Environment Variables

Create `.env` files in both frontend and backend directories:

### Backend (.env)

```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SECRET_KEY=your_secret_key
```

### Frontend (.env)

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_SUPABASE_URL=your_supabase_url
REACT_APP_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-chatbot
```

### 2. Backend Setup

````bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
````

### 4. Database Setup

- Create a Supabase project
- Run the provided SQL migrations
- Configure authentication providers

## 🛠️ Build Process

### Backend Build

We provide comprehensive build scripts to ensure code quality and readiness for deployment:

#### Option 1: Python Build Script (Recommended)

```bash
cd ai-chatbot-be
python3 build.py
```

#### Option 2: Shell Build Script (Quick)

```bash
cd ai-chatbot-be
./build.sh
```

**What the backend build does:**

- ✅ Python version check (3.8+)
- ✅ Dependencies verification
- ✅ Environment configuration validation
- ✅ Code formatting with Black (line length: 100)
- ✅ Syntax validation
- ✅ Import testing (enhanced RAG components)
- ✅ Code linting (flake8/pylint if available)
- ✅ Test execution (pytest if available)
- ✅ Application startup test

#### Build Output Example

```
============================================
           AI CHATBOT BACKEND BUILD
============================================

🔧 Checking Python version...
✅ Python 3.10.12 is compatible

🔧 Checking dependencies...
✅ All required dependencies are installed

🔧 Formatting code with Black...
✅ Code formatted with Black

🔧 Checking Python syntax...
✅ Syntax check passed for 25 files

🔧 Testing critical imports...
✅ All critical imports successful

============================================
                BUILD SUMMARY
============================================

🎉 BUILD SUCCESSFUL! All checks passed.
Your backend is ready for deployment.

Build completed in 4.23 seconds
```

### Frontend Build

```bash
cd ai-chatbot-fe

# Type checking
npm run type-check

# Linting
npm run lint

# Format code
npm run format

# Run all checks
npm run check-all

# Build for production
npm run build
```

### Development Workflow

#### Quick Development Build

```bash
# Backend quick build
cd ai-chatbot-be && ./build.sh

# Frontend quick check
cd ai-chatbot-fe && npm run format && npm run lint
```

#### Pre-deployment Build

```bash
# Comprehensive backend build
cd ai-chatbot-be && python3 build.py

# Comprehensive frontend build
cd ai-chatbot-fe && npm run check-all && npm run build
```

## Running the Application

### Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Start Frontend

```bash
cd frontend
npm start
```

The application will be available at:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Endpoints

### Authentication

- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/logout` - User logout

### Documents

- `POST /documents/upload` - Upload document
- `GET /documents` - List user documents
- `DELETE /documents/{id}` - Delete document

### Chat

- `POST /chat/message` - Send chat message
- `GET /chat/history` - Get chat history
- `WebSocket /chat/ws` - Real-time chat

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue on GitHub or contact the development team.
