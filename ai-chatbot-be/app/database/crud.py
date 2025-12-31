"""
Database CRUD operations with a Supabase-like interface.
This module provides a compatibility layer that mimics the Supabase client API.
"""
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import uuid
from app.database.connection import SessionLocal
from app.database.models import User, OTP, Document, DocumentChunk, ChatHistory, Meeting, UserGoogleAuth, CalendarEvent


class QueryResult:
    """Mimics Supabase query result"""
    def __init__(self, data: List[Dict]):
        self.data = data


class QueryBuilder:
    """Mimics Supabase query builder interface"""

    def __init__(self, model_class):
        self.model_class = model_class
        self._filters = []
        self._select_columns = "*"
        self._order_by = None
        self._limit_val = None
        self._in_filters = []

    def _get_session(self) -> Session:
        """Get a fresh session for each operation"""
        return SessionLocal()

    def select(self, columns: str = "*") -> "QueryBuilder":
        self._select_columns = columns
        return self

    def eq(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, "=", value))
        return self

    def neq(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, "!=", value))
        return self

    def gte(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, ">=", value))
        return self

    def lte(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, "<=", value))
        return self

    def gt(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, ">", value))
        return self

    def lt(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, "<", value))
        return self

    def in_(self, column: str, values: List[Any]) -> "QueryBuilder":
        self._in_filters.append((column, values))
        return self

    def order(self, column: str, desc: bool = False) -> "QueryBuilder":
        self._order_by = (column, desc)
        return self

    def limit(self, count: int) -> "QueryBuilder":
        self._limit_val = count
        return self

    def _convert_uuid(self, value: Any) -> Any:
        """Convert string to UUID if applicable"""
        if isinstance(value, str):
            try:
                return uuid.UUID(value)
            except ValueError:
                pass
        return value

    def _apply_filters(self, query, session: Session):
        for column, op, value in self._filters:
            col_attr = getattr(self.model_class, column, None)
            if col_attr is not None:
                if column in ('id', 'user_id', 'document_id'):
                    value = self._convert_uuid(value)

                if op == "=":
                    query = query.filter(col_attr == value)
                elif op == "!=":
                    query = query.filter(col_attr != value)
                elif op == ">=":
                    query = query.filter(col_attr >= value)
                elif op == "<=":
                    query = query.filter(col_attr <= value)
                elif op == ">":
                    query = query.filter(col_attr > value)
                elif op == "<":
                    query = query.filter(col_attr < value)

        for column, values in self._in_filters:
            col_attr = getattr(self.model_class, column, None)
            if col_attr is not None:
                if column in ('id', 'user_id', 'document_id'):
                    values = [self._convert_uuid(v) for v in values]
                query = query.filter(col_attr.in_(values))

        return query

    def execute(self) -> QueryResult:
        session = self._get_session()
        try:
            query = session.query(self.model_class)
            query = self._apply_filters(query, session)

            if self._order_by:
                col_attr = getattr(self.model_class, self._order_by[0], None)
                if col_attr is not None:
                    if self._order_by[1]:
                        query = query.order_by(col_attr.desc())
                    else:
                        query = query.order_by(col_attr.asc())

            if self._limit_val:
                query = query.limit(self._limit_val)

            results = query.all()
            data = [r.to_dict() for r in results]
            return QueryResult(data)
        finally:
            session.close()

    def update(self, data: Dict[str, Any]) -> QueryResult:
        session = self._get_session()
        try:
            query = session.query(self.model_class)
            query = self._apply_filters(query, session)

            results = query.all()
            updated_data = []

            for obj in results:
                for key, value in data.items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                session.add(obj)
                updated_data.append(obj.to_dict())

            session.commit()
            return QueryResult(updated_data)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete(self) -> QueryResult:
        session = self._get_session()
        try:
            query = session.query(self.model_class)
            query = self._apply_filters(query, session)

            results = query.all()
            deleted_data = [r.to_dict() for r in results]

            for obj in results:
                session.delete(obj)

            session.commit()
            return QueryResult(deleted_data)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


class InsertQueryBuilder(QueryBuilder):
    """Query builder for insert operations - supports single and bulk inserts"""

    def __init__(self, model_class, insert_data):
        super().__init__(model_class)
        # Handle both single dict and list of dicts
        if isinstance(insert_data, list):
            self._insert_data = insert_data
            self._is_bulk = True
        else:
            self._insert_data = [insert_data]
            self._is_bulk = False

    def _prepare_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a single record for insertion."""
        data = data.copy()

        # Handle UUID fields
        for key in ['id', 'user_id', 'document_id']:
            if key in data:
                data[key] = self._convert_uuid(data[key])

        # Handle document_ids array - convert list of strings to list of UUIDs
        if 'document_ids' in data and data['document_ids']:
            data['document_ids'] = [self._convert_uuid(doc_id) for doc_id in data['document_ids']]

        # Handle reserved field name mappings
        if self.model_class.__tablename__ == 'document_chunks' and 'metadata' in data:
            # 'metadata' is reserved in SQLAlchemy, so map to 'chunk_metadata'
            data['chunk_metadata'] = data.pop('metadata')
        if self.model_class.__tablename__ == 'chat_history' and 'model_config' in data:
            data['model_config_str'] = data.pop('model_config')

        return data

    def execute(self) -> QueryResult:
        session = self._get_session()
        try:
            results = []
            for record in self._insert_data:
                data = self._prepare_record(record)
                obj = self.model_class(**data)
                session.add(obj)
                results.append(obj)

            session.commit()

            # Refresh all objects to get generated values
            for obj in results:
                session.refresh(obj)

            return QueryResult([obj.to_dict() for obj in results])
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


class UpdateQueryBuilder(QueryBuilder):
    """Query builder for update operations"""

    def __init__(self, model_class, update_data: Dict[str, Any]):
        super().__init__(model_class)
        self._update_data = update_data

    def execute(self) -> QueryResult:
        return self.update(self._update_data)


class DeleteQueryBuilder(QueryBuilder):
    """Query builder for delete operations"""

    def execute(self) -> QueryResult:
        return self.delete()


class TableProxy:
    """Proxy class for table operations"""

    def __init__(self, model_class):
        self.model_class = model_class

    def select(self, columns: str = "*") -> QueryBuilder:
        return QueryBuilder(self.model_class).select(columns)

    def insert(self, data) -> InsertQueryBuilder:
        """Insert single record (dict) or multiple records (list of dicts)."""
        return InsertQueryBuilder(self.model_class, data)

    def update(self, data: Dict[str, Any]) -> UpdateQueryBuilder:
        return UpdateQueryBuilder(self.model_class, data)

    def delete(self) -> DeleteQueryBuilder:
        return DeleteQueryBuilder(self.model_class)


class RPCResult:
    """Result from RPC call"""
    def __init__(self, data: List[Dict]):
        self.data = data

    def execute(self) -> "RPCResult":
        """Execute is a no-op since we already executed in rpc()"""
        return self


class RPCBuilder:
    """Builder for RPC calls to PostgreSQL functions"""

    def __init__(self, function_name: str, params: Dict[str, Any]):
        self.function_name = function_name
        self.params = params

    def execute(self) -> RPCResult:
        from sqlalchemy import text
        session = SessionLocal()
        try:
            if self.function_name == "match_documents":
                # Call the match_documents function
                query_embedding = self.params.get("query_embedding", [])
                match_threshold = self.params.get("match_threshold", 0.5)
                match_count = self.params.get("match_count", 10)

                # Convert embedding list to PostgreSQL vector format string
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                # Use raw SQL with string formatting for the vector (SQLAlchemy doesn't handle ::vector cast well)
                sql = f"""
                    SELECT id, document_id, content, metadata, similarity
                    FROM match_documents(
                        '{embedding_str}'::vector,
                        {match_threshold},
                        {match_count}
                    )
                """
                result = session.execute(text(sql)).fetchall()

                data = [
                    {
                        "id": str(row[0]),
                        "document_id": str(row[1]),
                        "content": row[2],
                        "metadata": row[3],
                        "similarity": float(row[4]) if row[4] else 0.0
                    }
                    for row in result
                ]
                return RPCResult(data)

            elif self.function_name == "match_user_documents":
                query_embedding = self.params.get("query_embedding", [])
                user_id = self.params.get("user_id_param")
                match_threshold = self.params.get("match_threshold", 0.5)
                match_count = self.params.get("match_count", 10)

                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                sql = f"""
                    SELECT id, document_id, content, metadata, similarity
                    FROM match_user_documents(
                        '{embedding_str}'::vector,
                        '{user_id}'::uuid,
                        {match_threshold},
                        {match_count}
                    )
                """
                result = session.execute(text(sql)).fetchall()

                data = [
                    {
                        "id": str(row[0]),
                        "document_id": str(row[1]),
                        "content": row[2],
                        "metadata": row[3],
                        "similarity": float(row[4]) if row[4] else 0.0
                    }
                    for row in result
                ]
                return RPCResult(data)

            else:
                raise ValueError(f"Unknown RPC function: {self.function_name}")

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"RPC error: {e}")
            return RPCResult([])
        finally:
            session.close()


class DatabaseClient:
    """Main database client that mimics Supabase client interface"""

    _tables = {
        "users": User,
        "otps": OTP,
        "documents": Document,
        "document_chunks": DocumentChunk,
        "chat_history": ChatHistory,
        "meetings": Meeting,
        "user_google_auth": UserGoogleAuth,
        "calendar_events": CalendarEvent,
    }

    def table(self, name: str) -> TableProxy:
        model_class = self._tables.get(name)
        if model_class is None:
            raise ValueError(f"Unknown table: {name}")
        return TableProxy(model_class)

    def rpc(self, function_name: str, params: Dict[str, Any] = None) -> RPCBuilder:
        """
        Call a PostgreSQL function (RPC).

        Args:
            function_name: Name of the function to call
            params: Parameters to pass to the function

        Returns:
            RPCBuilder that can be executed
        """
        return RPCBuilder(function_name, params or {})


# Global database client instance
db_client = DatabaseClient()
