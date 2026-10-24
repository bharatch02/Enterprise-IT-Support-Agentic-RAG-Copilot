const DEFAULT_TIMEOUT_MS = 120_000;

export async function fetchJson(
  url,
  options = {},
  { timeoutMs = DEFAULT_TIMEOUT_MS, fetchImpl = globalThis.fetch } = {},
) {
  const controller = new AbortController();
  let timeoutExpired = false;
  const timeoutId = setTimeout(() => {
    timeoutExpired = true;
    controller.abort();
  }, timeoutMs);
  let requestId = "";
  const callerSignal = options.signal;
  const requestOptions = { ...options };
  delete requestOptions.signal;
  const abortFromCaller = () => controller.abort(callerSignal.reason);
  callerSignal?.addEventListener("abort", abortFromCaller, { once: true });

  if (callerSignal?.aborted) {
    abortFromCaller();
  }

  const withRequestId = (message) =>
    requestId && !message.includes(requestId)
      ? `${message} (Request ID: ${requestId})`
      : message;

  try {
    const response = await fetchImpl(url, {
      ...requestOptions,
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
          const message = response.status === 502
            ? "The support service returned HTTP 502 and may have restarted or be temporarily unavailable. Check Render logs and try again."
            : `Request failed with HTTP ${response.status}.`;
          throw new Error(withRequestId(message));
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
      const fallbackMessage = response.status === 502
        ? "The support service returned HTTP 502 and may have restarted or be temporarily unavailable. Check Render logs and try again."
        : `Request failed with HTTP ${response.status}.`;
      throw new Error(
        withRequestId(detail || fallbackMessage),
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
      if (callerSignal?.aborted && !timeoutExpired) {
        throw new Error(withRequestId("The request was cancelled."));
      }
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
    callerSignal?.removeEventListener("abort", abortFromCaller);
  }
}
