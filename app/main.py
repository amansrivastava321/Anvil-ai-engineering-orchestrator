"""
FastAPI application entry point for the Local AI Engineering Orchestrator.

This is the main server that:
- Exposes REST API endpoints for agent interactions
- Manages application lifecycle (startup/shutdown)
- Initializes all services and integrations
- Handles CORS, middleware, and routing
- Provides health check and monitoring endpoints
- Starts background intelligence update services
"""

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config.settings import settings, validate_settings_on_startup
from app.core.monitoring.logging import (
    setup_logging,
    get_logger,
    RequestLogger,
)
from app.core.monitoring.metrics import setup_metrics

# Initialize logging first
setup_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    validate_settings_on_startup()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade local AI engineering orchestrator with Graphify and skillfile integration",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=settings.security.cors_methods,
        allow_headers=settings.security.cors_headers,
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        async with RequestLogger(request):
            response = await call_next(request)
            return response
    
    # Prometheus metrics
    if settings.monitoring.enable_prometheus:
        setup_metrics(app)
    
    # Dashboard (served at /dashboard)
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="dashboard")

    # Register routes and lifecycle events
    register_routes(app)
    register_lifecycle_events(app)
    
    logger.info(
        "Application created",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
    )
    
    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    from app.api.v1.router import api_router
    
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    
    @app.get("/")
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
            "docs": "/docs" if settings.debug else "disabled",
            "health": "/health",
            "api": settings.api_v1_prefix,
        }
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
    
    @app.get("/health/detailed")
    async def detailed_health():
        try:
            from app.services.orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            return await orchestrator.health_check()
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
    
    from app.utils.validators import PathSecurityError

    @app.exception_handler(PathSecurityError)
    async def path_security_exception_handler(request: Request, exc: PathSecurityError):
        logger.warning(
            "Path security violation",
            error=str(exc),
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "status_code": 400},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(
            "Validation error",
            error=str(exc),
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "status_code": 400},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=str(exc.detail),
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code},
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            error=str(exc),
            path=str(request.url.path),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else None,
            },
        )


def register_lifecycle_events(app: FastAPI) -> None:
    """Register application lifecycle event handlers."""
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("Starting application services...")
        
        # Check Graphify
        try:
            from app.integrations.graphify.parser import GraphifyWrapper
            wrapper = GraphifyWrapper(str(Path.cwd()))
            if wrapper.is_installed():
                logger.info("Graphify is installed")
            else:
                logger.warning(
                    "Graphify not installed. Install with: "
                    "pip install graphifyy && graphify install"
                )
        except Exception as e:
            logger.warning(f"Could not check Graphify: {e}")
        
        # Check skillfile
        try:
            from app.integrations.skillfile.client import get_skillfile_client
            skillfile = get_skillfile_client()
            if skillfile.is_installed:
                logger.info("skillfile is installed")
            else:
                logger.warning(
                    "skillfile not installed. Install with: "
                    "curl -fsSL https://github.com/eljulians/skillfile/releases/latest/download/install.sh | sh"
                )
        except Exception as e:
            logger.warning(f"Could not check skillfile: {e}")
        
        # Check Ollama
        try:
            from app.services.orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            health = await orchestrator.ollama.health_check()
            if health.get("status") == "healthy":
                logger.info("Ollama is connected")
            else:
                logger.warning("Ollama connection issue")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
        
        # Start evolution scheduler if enabled
        if settings.feature_evolution_enabled:
            try:
                from app.services.evolution_service import get_evolution_service
                evolution_svc = get_evolution_service()
                asyncio.create_task(
                    evolution_svc.schedule_evolution_cycles(
                        interval_hours=settings.evolution_cycle_interval_hours
                    ),
                    name="evolution_scheduler",
                )
                logger.info(
                    "Evolution scheduler started",
                    interval_hours=settings.evolution_cycle_interval_hours,
                )
            except Exception as e:
                logger.warning(f"Could not start evolution scheduler: {e}")

        # Start proactive monitoring if enabled
        if settings.feature_proactive_monitoring:
            try:
                from app.services.proactive_service import get_proactive_service
                proactive = get_proactive_service()
                if settings.monitored_repositories:
                    result = await proactive.start_watching(
                        settings.monitored_repositories,
                        poll_interval=settings.monitor_poll_interval,
                    )
                    logger.info(
                        "Proactive monitoring started",
                        started=result.get("started"),
                        total=result.get("total_watching"),
                    )
                else:
                    logger.info(
                        "Proactive monitoring enabled but no repositories configured "
                        "(set MONITORED_REPOSITORIES env var)"
                    )
            except Exception as e:
                logger.warning(f"Could not start proactive monitoring: {e}")

        logger.info("Application startup complete")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down application...")

        # Stop proactive monitoring first (graceful task cancellation)
        if settings.feature_proactive_monitoring:
            try:
                from app.services.proactive_service import get_proactive_service
                proactive = get_proactive_service()
                if proactive.is_running:
                    await proactive.stop_watching()
                    logger.info("Proactive monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping proactive monitoring: {e}")

        try:
            from app.services.orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            await orchestrator.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        logger.info("Application shut down complete")


# Create the application instance
app = create_app()


def run_server():
    """Entry point for running the server."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if not settings.debug else 1,
        reload=settings.debug,
        log_level=settings.monitoring.log_level.value.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()