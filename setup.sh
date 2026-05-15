mkdir -p app/api/v1/endpoints
mkdir -p app/core/{config,security,monitoring,cache}
mkdir -p app/services
mkdir -p app/models/repository
mkdir -p app/agents/{base,specialized}
mkdir -p app/tools/{file_system,code_analysis,testing}
mkdir -p app/integrations/{graphify,ollama,ide}

mkdir -p config/{environments,models}

mkdir -p data/{graphify_out,artifacts,logs,metrics,cache}

mkdir -p tests/{unit,integration,e2e,fixtures}

mkdir -p scripts/{deployment,monitoring,migration}

mkdir -p docs/{api,architecture,deployment}

mkdir -p .github/workflows

touch app/__init__.py
touch app/api/__init__.py
touch app/api/v1/__init__.py
touch app/api/v1/endpoints/__init__.py
touch app/core/__init__.py
touch app/core/config/__init__.py
touch app/core/security/__init__.py
touch app/core/monitoring/__init__.py
touch app/core/cache/__init__.py
touch app/services/__init__.py
touch app/models/__init__.py
touch app/models/repository/__init__.py
touch app/agents/__init__.py
touch app/agents/base/__init__.py
touch app/agents/specialized/__init__.py
touch app/tools/__init__.py
touch app/tools/file_system/__init__.py
touch app/tools/code_analysis/__init__.py
touch app/tools/testing/__init__.py
touch app/integrations/__init__.py
touch app/integrations/graphify/__init__.py
touch app/integrations/ollama/__init__.py
touch app/integrations/ide/__init__.py

touch app/main.py

touch app/api/v1/router.py
touch app/api/v1/endpoints/agent.py
touch app/api/v1/endpoints/models.py
touch app/api/v1/endpoints/health.py
