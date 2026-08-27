// Rust SDK for Vsector — reqwest + serde
use serde_json::Value;

pub struct Client {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self { base_url: base_url.into(), api_key: api_key.into(), http: reqwest::Client::new() }
    }
    pub async fn upsert(&self, namespace: &str, vectors: Value) -> anyhow::Result<Value> {
        let body = serde_json::json!({"namespace": namespace, "vectors": vectors});
        let resp = self.http.post(format!("{}/v1/vectors/upsert", self.base_url))
            .header("X-API-Key", &self.apiKey())
            .json(&body).send().await?.json().await?;
        Ok(resp)
    }
    pub async fn query(&self, namespace: &str, vector: Vec<f32>, top_k: usize) -> anyhow::Result<Value> {
        let body = serde_json::json!({"namespace": namespace, "vector": vector, "top_k": top_k});
        let resp = self.http.post(format!("{}/v1/vectors/query", self.base_url))
            .header("X-API-Key", &self.apiKey())
            .json(&body).send().await?.json().await?;
        Ok(resp)
    }
    fn apiKey(&self) -> String { self.api_key.clone() }
}
