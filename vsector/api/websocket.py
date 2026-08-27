"""WebSocket & GraphQL stubs for Client/External Systems layer."""
from fastapi import WebSocket, WebSocketDisconnect
import json

# WebSocket: real-time vector query subscription
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
                # Echo stub - in prod delegate to QueryEngine
                await websocket.send_text(json.dumps({"echo": req, "results": []}))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass

# GraphQL stub (use strawberry or ariadne in prod)
# schema.graphql:
# type Query { query(namespace: String!, vector: [Float!]!, topK: Int): [Result] }
# type Mutation { upsert(namespace: String!, vectors: [VectorInput!]!): UpsertPayload }
GRAPHQL_SDL = """
type Vector { id: ID! namespace: String! values: [Float!]! metadata: JSON }
type QueryResult { id: ID! score: Float! metadata: JSON }
type Query {
  query(namespace: String!, vector: [Float!]!, topK: Int = 10, efSearch: Int): [QueryResult!]!
  fetch(namespace: String!, ids: [ID!]!): [Vector!]!
}
type Mutation {
  upsert(namespace: String!, vectors: [VectorInput!]!): UpsertResult!
  delete(namespace: String!, ids: [ID!]!): DeleteResult!
}
input VectorInput { id: ID vector: [Float!]! metadata: JSON tags: [String!] }
type UpsertResult { upserted: Int! ids: [ID!]! }
type DeleteResult { deleted: Int! }
scalar JSON
"""
