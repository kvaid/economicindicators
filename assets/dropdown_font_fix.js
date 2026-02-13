(function () {
  function setFont(el) {
    if (!el) return;
    el.style.setProperty("font-size", "0.72rem", "important");
  }

  function applyDropdownFontFix() {
    // Closed-state selected value/placeholder/input for credit spread baseline dropdown.
    const baselineRoot = document.getElementById("cs-baseline-tenor");
    if (baselineRoot) {
      setFont(baselineRoot);
      baselineRoot
        .querySelectorAll(
          "[class*='singleValue'], [class*='placeholder'], [class*='value'], [class*='control'], input, div[role='combobox'], span"
        )
        .forEach(setFont);
    }

    // Opened menu options (react-select renders these dynamically/possibly in portals).
    document.querySelectorAll("div[role='option']").forEach(setFont);
    document.querySelectorAll("div[role='listbox']").forEach(setFont);
  }

  document.addEventListener("DOMContentLoaded", applyDropdownFontFix);
  document.addEventListener("click", applyDropdownFontFix, true);
  document.addEventListener("keyup", applyDropdownFontFix, true);

  const observer = new MutationObserver(applyDropdownFontFix);
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
