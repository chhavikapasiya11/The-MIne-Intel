// nlp.js
// Extract numeric mining parameters from a text snippet in React JS

export const FIELD_SYNONYMS = {
  CMRR: [
    'cmrr','cmr','cr','roof rating','mass rate','rock rate','rating','r rating','geo rate','stability','cmrr value'
  ],
  PRSUP: [
    'prsup','support load','roof support','ps','support','load','roof load','r load','sup load','press','pressure','p support','r support'
  ],
  depth_of_cover: [
    'depth of cover','overburden','cover depth','depth','cover','burden','ob','d cover','deep','top depth','surface depth','dc','cd'
  ],
  intersection_diagonal: [
    'intersection diagonal','diagonal','crosscut diagonal','diag','id','int diag','cross diag','x diag','inter diag','d length','junction d','crosscut d'
  ],
  mining_height: [
    'mining height','seam height','extraction height','mh','height','h','mine h','seam h','ex h','work h','coal h','m height','face h'
  ]
};

// helper: extract first number from string
export function extractNumber(snippet) {
  const match = snippet.match(/(-?\d+(?:\.\d+)?)/);
  if (!match) return null;
  const value = parseFloat(match[1]);
  return isNaN(value) ? null : value;
}

// main function: extract features from text
export function extractFeaturesFromText(text) {
  const lowered = text.toLowerCase();
  const extracted = {
    CMRR: null,
    PRSUP: null,
    depth_of_cover: null,
    intersection_diagonal: null,
    mining_height: null
  };

  Object.entries(FIELD_SYNONYMS).forEach(([field, synonyms]) => {
    for (const synonym of synonyms) {
      const escaped = synonym.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const pattern = new RegExp(
        `${escaped}[^0-9-]*(-?\\d+(?:\\.\\d+)?)(?:\\s?(?:meters|meter|m|percent|%))?`,
        'i'
      );
      const match = lowered.match(pattern);
      if (match) {
        extracted[field] = parseFloat(match[1]);
        break;
      }
    }

    if (extracted[field] !== null) return;

    // alternative pattern: "set synonym to number" or "synonym is number"
    for (const synonym of synonyms) {
      const escaped = synonym.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const altPattern = new RegExp(
        `(?:set|make|is|at)\\s+${escaped}[^0-9-]*(-?\\d+(?:\\.\\d+)?)`,
        'i'
      );
      const match = lowered.match(altPattern);
      if (match) {
        extracted[field] = parseFloat(match[1]);
        break;
      }
    }
  });

  return extracted;
}
