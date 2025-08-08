// assets/clientside.js
console.log("✅ clientside.js loaded");
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    clearTooltipOnMouseLeave: function (hoverDataList) {
      let allNone = true;
      for (let hd of hoverDataList) {
        if (hd && hd.points && hd.points.length > 0) {
          allNone = false;
          break;
        }
      }
      if (allNone) {
        return { display: "none" };
      }
      return window.dash_clientside.no_update;
    }
  }
});

