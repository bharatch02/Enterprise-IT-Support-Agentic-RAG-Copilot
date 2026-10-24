const DEFAULT_TIMEOUT_MS = 120_000;

export async function fetchJson(
  url,
  options = {},
  { timeoutMs = DEFAULT_TIMEOUT_MS, fetchImpl = globalThis.fetch } = {},
) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  let requestId = "";

  const withRequestId = (message) =>
    requestId && !message.includes(requestId)
      ? `${message} (Request ID: ${requestId})`
      : message;

  try {
    const response = await fetchImpl(url, {
      ...options,
      signal: controller.signal,
    });
    requestId = response.headers?.get?.("x-request-id") || "";
    const responseText = await response.text();
    let data = null;

    if (responseText.trim()) {
      try {
        data = JSON.parse(responseText);
      } catch {
        if (!response.ok) {
          throw new Error(withRequestId(`Request failed with HTTP ${response.status}.`));
        }

        throw new Error(
          withRequestId("The server returned an invalid response. Please try again."),
        );
      }
    }

    if (!response.ok) {
      const detail =
        data && typeof data === "object" && !Array.isArray(data)
          ? data.detail || data.message
          : "";
      throw new Error(
        withRequestId(detail || `Request failed with HTTP ${response.status}.`),
      );
    }

    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new Error(
        withRequestId(
          responseText.trim()
            ? "The server returned an unexpected response. Please try again."
            : "The server returned an empty response. Check the service logs and try again.",
        ),
      );
    }

    return data;
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(
        withRequestId(
          `The request timed out after ${Math.ceil(timeoutMs / 1000)} seconds. Please try again.`,
        ),
      );
    }

    if (error instanceof TypeError) {
      throw new Error(
        withRequestId(
          "Could not reach the support service. Check your connection and try again.",
        ),
      );
    }

    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}
