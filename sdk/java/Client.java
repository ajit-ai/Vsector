// Java SDK for Vsector — Java 17+ (HttpClient)
package io.vsector;

import java.net.http.*;
import java.net.URI;
import java.util.*;

public class Client {
    private final String baseUrl;
    private final String apiKey;
    private final HttpClient http = HttpClient.newHttpClient();

    public Client(String baseUrl, String apiKey) {
        this.baseUrl = baseUrl; this.apiKey = apiKey;
    }

    public String upsert(String namespace, String vectorsJson) throws Exception {
        var req = HttpRequest.newBuilder(URI.create(baseUrl + "/v1/vectors/upsert"))
            .header("Content-Type", "application/json")
            .header("X-API-Key", apiKey)
            .POST(HttpRequest.BodyPublishers.ofString("{\"namespace\":\""+namespace+"\",\"vectors\":"+vectorsJson+"}"))
            .build();
        return http.send(req, HttpResponse.BodyHandlers.ofString()).body();
    }

    public String query(String namespace, String vectorJson, int topK) throws Exception {
        var body = String.format("{\"namespace\":\"%s\",\"vector\":%s,\"top_k\":%d}", namespace, vectorJson, topK);
        var req = HttpRequest.newBuilder(URI.create(baseUrl + "/v1/vectors/query"))
            .header("Content-Type", "application/json")
            .header("X-API-Key", apiKey)
            .POST(HttpRequest.BodyPublishers.ofString(body)).build();
        return http.send(req, HttpResponse.BodyHandlers.ofString()).body();
    }
}
