import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement Element.scrollTo (real browsers do) -- a
// well-known jsdom gap, not something AgentFeed's auto-scroll code needs
// to guard against for real usage. Polyfilled here, in test setup only.
if (typeof Element.prototype.scrollTo !== "function") {
  Element.prototype.scrollTo = () => {};
}
