"""6C: Event Streaming (Apache Kafka Integration)

Producer: publishes vector upsert/delete events.
Consumer: ingests from Kafka (KafkaConsumer flow: Client -> API Gateway -> Kafka -> Ingest Service).
Topic naming: vsector.<namespace>.<event> (e.g. vsector.products.upsert)
"""
from __future__ import annotations

import json
import logging
import asyncio
from typing import AsyncIterable

logger = logging.getLogger(__name__)

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer  # type: ignore
    HAS_KAFKA = True
except Exception:
    HAS_KAFKA = False


class EventBus:
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic_prefix: str = "vsector."):
        self.bootstrap = bootstrap_servers
        self.prefix = topic_prefix
        self._producer = None

    def topic_for(self, namespace: str, event: str) -> str:
        return f"{self.prefix}{namespace}.{event}"

    async def start(self):
        if not HAS_KAFKA:
            logger.warning("aiokafka not installed - EventBus running in no-op mode")
            return
        self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap, value_serializer=lambda v: json.dumps(v).encode())
        await self._producer.start()
        logger.info(f"Kafka producer started {self.bootstrap}")

    async def stop(self):
        if self._producer:
            await self._producer.stop()

    async def publish(self, namespace: str, event: str, payload: dict) -> None:
        topic = self.topic_for(namespace, event)
        if HAS_KAFKA and self._producer:
            await self._producer.send_and_wait(topic, payload)
        else:
            logger.debug(f"[EventBus no-op] {topic} -> {payload}")

    async def consume(self, namespace: str, event: str, ingest_service) -> None:
        """Kafka Consumer -> Ingest Service loop."""
        if not HAS_KAFKA:
            logger.warning("Kafka consumer disabled - aiokafka missing")
            return
        topic = self.topic_for(namespace, event)
        consumer = AIOKafkaConsumer(topic, bootstrap_servers=self.bootstrap, value_deserializer=lambda v: json.loads(v.decode()), group_id=f"vsector-{namespace}-consumer")
        await consumer.start()
        try:
            async for msg in consumer:
                try:
                    payload = msg.value
                    if event == "upsert":
                        await ingest_service.upsert(namespace, payload.get("vectors", []))
                    elif event == "delete":
                        await ingest_service.delete(namespace, ids=payload.get("ids"))
                    logger.info(f"Consumed {topic} offset {msg.offset}")
                except Exception as e:
                    logger.error(f"Consumer error: {e}")
        finally:
            await consumer.stop()
