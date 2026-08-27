// Go SDK for Vsector — mirrors Python SDK + gRPC
package vsector

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type Client struct {
	BaseURL string
	APIKey  string
	HTTP    *http.Client
}

func New(baseURL, apiKey string) *Client {
	return &Client{BaseURL: baseURL, APIKey: apiKey, HTTP: &http.Client{}}
}

func (c *Client) do(method, path string, body any, out any) error {
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest(method, c.BaseURL+path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("X-API-Key", c.APIKey)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("http %d", resp.StatusCode)
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func (c *Client) CreateNamespace(name string, dim int) (map[string]any, error) {
	var out map[string]any
	err := c.do("POST", "/v1/namespaces", map[string]any{"name": name, "dimension": dim}, &out)
	return out, err
}

func (c *Client) Upsert(namespace string, vectors []map[string]any) (map[string]any, error) {
	var out map[string]any
	err := c.do("POST", "/v1/vectors/upsert", map[string]any{"namespace": namespace, "vectors": vectors}, &out)
	return out, err
}

func (c *Client) Query(namespace string, vector []float32, topK int) (map[string]any, error) {
	var out map[string]any
	err := c.do("POST", "/v1/vectors/query", map[string]any{"namespace": namespace, "vector": vector, "top_k": topK}, &out)
	return out, err
}
