// Page Search functionality
let pageSearchMatches = [];
let pageSearchCurrentIndex = -1;
let originalNodes = new Map(); // Store original node content for clearing

function clearPageSearchHighlights() {
  // Restore original text nodes
  for (const [span, { parent, originalContent }] of originalNodes) {
    if (span.parentNode === parent) { // Check if parent still exists
      const textNode = document.createTextNode(originalContent);
      parent.replaceChild(textNode, span);
    }
  }
  // Clear previous highlights for spans that might have been created
  // without replacing a full node (less likely with new logic but good practice)
  document.querySelectorAll('.page-search-highlight').forEach(span => {
      if (span.parentNode) {
          const textNode = document.createTextNode(span.textContent);
          span.parentNode.replaceChild(textNode, span);
      }
  });

  pageSearchMatches = [];
  pageSearchCurrentIndex = -1;
  originalNodes.clear();
  updateMatchCount();
}

function pageSearch() {
  console.log("pageSearch started"); // DEBUG
  clearPageSearchHighlights(); // Clear previous results first

  const pattern = document.getElementById('page-search-input').value;
  const caseSensitive = document.getElementById('case-sensitive-check').checked;
  const isRegex = document.getElementById('regex-check').checked;
  if (!pattern) { 
      console.log("pageSearch: No pattern provided."); // DEBUG
      updateMatchCount(); 
      return false; 
  } 
  console.log(`pageSearch: Searching for "${pattern}", caseSensitive: ${caseSensitive}, isRegex: ${isRegex}`); // DEBUG

  const flags = caseSensitive ? 'g' : 'gi';
  let regex;
  try {
    regex = isRegex ? new RegExp(pattern, flags)
                    : new RegExp(pattern.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'), flags);
  } catch (e) {
    console.error("pageSearch: Invalid regex:", e); // DEBUG
    alert('Invalid regular expression: ' + e.message);
    return false; 
  }

  // Look for content area - try multiple selectors for different templates
  let contentArea = document.querySelector('.content');
  if (!contentArea) {
    // Try alternative containers
    contentArea = document.querySelector('.container');
    if (!contentArea) {
      contentArea = document.body; // Fallback to body if no container found
    }
  }
  
  console.log("pageSearch: Content area found.", contentArea); // DEBUG

  const nodesToProcess = [];
  // Wrap main logic in try...catch
  try {
      const walker = document.createTreeWalker(contentArea, NodeFilter.SHOW_TEXT, null, false);
      let node;
      console.log("pageSearch: Starting TreeWalker..."); // DEBUG
      while ((node = walker.nextNode())) {
          // Skip nodes within the search form itself or irrelevant tags
          if (node.parentNode?.closest && (node.parentNode.closest('#page-search-form') || node.parentNode.closest('SCRIPT, STYLE, NOSCRIPT, BUTTON, INPUT, TEXTAREA, SELECT'))) {
              continue;
          }

          const text = node.nodeValue;
          if (text && text.trim() !== '') { // Ensure node has text content
              regex.lastIndex = 0; // Reset regex state for each node
              if (regex.test(text)) { 
                  nodesToProcess.push(node);
                  // Store original content *before* modification if not already stored
                  if (!originalNodes.has(node)) {
                      originalNodes.set(node, { parent: node.parentNode, originalContent: node.nodeValue });
                  }
              }
          }
      }
      console.log(`pageSearch: Found ${nodesToProcess.length} text nodes with potential matches.`); // DEBUG

      // Phase 2: Apply highlighting 
      nodesToProcess.forEach((node, nodeIndex) => {
          if (!node || !node.parentNode) {
              console.warn(`pageSearch: Node at index ${nodeIndex} is invalid or detached, skipping.`); // DEBUG
              return; // Skip processing if node is invalid
          }
          const text = node.nodeValue; 
          const frag = document.createDocumentFragment();
          let lastIndex = 0;
          let match;
          regex.lastIndex = 0; 

          while ((match = regex.exec(text))) {
              const before = text.slice(lastIndex, match.index);
              if (before) frag.appendChild(document.createTextNode(before));

              const span = document.createElement('span');
              span.className = 'page-search-highlight';
              span.textContent = match[0];
              frag.appendChild(span);
              pageSearchMatches.push(span); 

              lastIndex = match.index + match[0].length;
          }

          const after = text.slice(lastIndex);
          if (after) frag.appendChild(document.createTextNode(after));
          
          // Safely replace node
          try {
              node.parentNode.replaceChild(frag, node);
          } catch (replaceError) {
               console.error(`pageSearch: Error replacing node at index ${nodeIndex}:`, replaceError, "Node:", node, "Parent:", node.parentNode); // DEBUG
          }
      });

  } catch (error) {
       console.error("pageSearch: Error during text node processing or highlighting:", error); // DEBUG
  }

  console.log(`pageSearch: Total matches found: ${pageSearchMatches.length}`); // DEBUG
  updateMatchCount();
  if (pageSearchMatches.length) {
    pageSearchCurrentIndex = 0;
    highlightCurrent();
  } else {
       console.log("pageSearch: No matches found."); // DEBUG
  }
  return false; // Prevent form submission
}

function updateMatchCount() {
  const countEl = document.getElementById('match-count');
  if (countEl) { 
      countEl.textContent = pageSearchMatches.length;
  } else {
      console.error("updateMatchCount: match-count element not found"); // DEBUG
  }
}

function highlightCurrent() {
   if (!pageSearchMatches.length || pageSearchCurrentIndex < 0 || pageSearchCurrentIndex >= pageSearchMatches.length) {
       console.log("highlightCurrent: No matches or invalid index.", pageSearchCurrentIndex); // DEBUG
       return;
   }
   console.log(`highlightCurrent: Highlighting index ${pageSearchCurrentIndex}`); // DEBUG
   
   pageSearchMatches.forEach((span, idx) => {
      // Use try-catch for safety as spans might become detached
      try {
          span.classList.toggle('current-match', idx === pageSearchCurrentIndex);
          if (idx === pageSearchCurrentIndex) {
              console.log("highlightCurrent: Scrolling to match:", span); // DEBUG
              // Check if element is visible before scrolling
              const rect = span.getBoundingClientRect();
               if (rect.width > 0 && rect.height > 0) {
                 span.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
               } else {
                  console.warn("highlightCurrent: Match span is not visible, skipping scrollIntoView.", span); // DEBUG
               }
          }
      } catch(e) {
          console.error(`highlightCurrent: Error processing span at index ${idx}:`, e); // DEBUG
      }
   });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Setup event listeners for search navigation
  const prevMatchBtn = document.getElementById('prev-match');
  const nextMatchBtn = document.getElementById('next-match');
  const searchForm = document.getElementById('page-search-form');
  
  if (searchForm) {
    searchForm.addEventListener('submit', function(e) {
      e.preventDefault();
      pageSearch();
    });
  } else {
    console.warn("Page search form not found in the document");
  }
  
  if (prevMatchBtn) {
    prevMatchBtn.addEventListener('click', function() {
      if (!pageSearchMatches.length) return;
      pageSearchCurrentIndex = (pageSearchCurrentIndex - 1 + pageSearchMatches.length) % pageSearchMatches.length;
      highlightCurrent();
    });
  }
  
  if (nextMatchBtn) {
    nextMatchBtn.addEventListener('click', function() {
      if (!pageSearchMatches.length) return;
      pageSearchCurrentIndex = (pageSearchCurrentIndex + 1) % pageSearchMatches.length;
      highlightCurrent();
    });
  }
}); 