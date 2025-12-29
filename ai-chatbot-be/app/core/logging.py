import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import traceback


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        if hasattr(record, 'method'):
            log_entry['method'] = record.method
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)


class AppLogger:
    """Centralized logging configuration"""
    
    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # JSON formatter
        json_formatter = JSONFormatter()
        
        # Console handler with JSON format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setFormatter(json_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]
    
    def log_request(self, logger: logging.Logger, method: str, endpoint: str, 
                   user_id: Optional[str] = None, request_id: Optional[str] = None):
        """Log incoming request"""
        logger.info(
            f"Incoming {method} request to {endpoint}",
            extra={
                "method": method,
                "endpoint": endpoint,
                "user_id": user_id,
                "request_id": request_id
            }
        )
    
    def log_response(self, logger: logging.Logger, method: str, endpoint: str,
                    status_code: int, execution_time: float,
                    user_id: Optional[str] = None, request_id: Optional[str] = None):
        """Log outgoing response"""
        logger.info(
            f"Response {status_code} for {method} {endpoint} in {execution_time:.3f}s",
            extra={
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "execution_time": execution_time,
                "user_id": user_id,
                "request_id": request_id
            }
        )
    
    def log_error(self, logger: logging.Logger, error: Exception, 
                 context: Optional[Dict[str, Any]] = None,
                 user_id: Optional[str] = None, request_id: Optional[str] = None):
        """Log error with context"""
        extra = {
            "error_type": error.__class__.__name__,
            "user_id": user_id,
            "request_id": request_id
        }
        if context:
            extra.update(context)
        
        logger.error(
            f"Error occurred: {str(error)}",
            extra=extra,
            exc_info=True
        )
    
    def log_database_operation(self, logger: logging.Logger, operation: str, 
                             table: str, success: bool, execution_time: float,
                             user_id: Optional[str] = None):
        """Log database operations"""
        level = logging.INFO if success else logging.ERROR
        logger.log(
            level,
            f"Database {operation} on {table}: {'SUCCESS' if success else 'FAILED'} ({execution_time:.3f}s)",
            extra={
                "operation": operation,
                "table": table,
                "success": success,
                "execution_time": execution_time,
                "user_id": user_id
            }
        )
    
    def log_external_api(self, logger: logging.Logger, service: str, endpoint: str,
                        status_code: int, execution_time: float,
                        user_id: Optional[str] = None):
        """Log external API calls"""
        logger.info(
            f"External API call to {service} {endpoint}: {status_code} ({execution_time:.3f}s)",
            extra={
                "service": service,
                "endpoint": endpoint,
                "status_code": status_code,
                "execution_time": execution_time,
                "user_id": user_id
            }
        )


# Global logger instance
app_logger = AppLogger()

# Convenience function to get logger
def get_logger(name: str) -> logging.Logger:
    return app_logger.get_logger(name)