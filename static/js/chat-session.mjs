export function createChatSession() {
  let generation = 0;
  let activeRequest = null;

  return {
    begin() {
      if (activeRequest) return null;

      const request = {
        generation,
        controller: new AbortController(),
      };
      activeRequest = request;
      return request;
    },

    isCurrent(request) {
      return request?.generation === generation;
    },

    finish(request) {
      if (activeRequest === request) activeRequest = null;
    },

    reset() {
      generation += 1;
      activeRequest?.controller.abort();
      activeRequest = null;
    },
  };
}
