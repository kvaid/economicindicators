(function () {
  function setFont(el) {
    if (!el) return;
    el.style.setProperty("font-size", "0.72rem", "important");
  }

  function setFontDeep(root) {
    if (!root) return;
    setFont(root);
    root.querySelectorAll("*").forEach(setFont);
  }

  function applyDateFontFix() {
    // Closed state: both DatePickerSingle inputs in left panel.
    document
      .querySelectorAll(
        "#start-date, #end-date, .date-range, #start-date input, #end-date input, .date-range input, .DateInput_input, .DateInput_input__small"
      )
      .forEach(setFontDeep);

    // Opened calendar popup text (react-dates + portal-rendered variants).
    document
      .querySelectorAll(
        ".SingleDatePicker_picker, .SingleDatePicker_picker__portal, .DateRangePicker_picker, .DayPicker, .CalendarMonth, .CalendarDay, [class*='DayPicker'], [class*='CalendarMonth'], [class*='CalendarDay']"
      )
      .forEach(setFontDeep);
  }

  document.addEventListener("DOMContentLoaded", applyDateFontFix);
  document.addEventListener("click", applyDateFontFix, true);
  document.addEventListener("keyup", applyDateFontFix, true);

  const observer = new MutationObserver(applyDateFontFix);
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
