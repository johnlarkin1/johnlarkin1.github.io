window.jQuery = window.$ = require("jquery");
require("velocity-animate/velocity.js");
require("lazysizes");
require("lazysizes/plugins/unveilhooks/ls.unveilhooks.js");
require("./pyodide-runner.js");
require("./vector-search.js");

// Jquery & Velocity JS included in GULP
$(document).ready(function () {
  // Reveal header once DOM is ready (prevents FOUC)
  $(".header").css("opacity", "1");

  toggleMobileNav();
  ShowHideNav();
  formCheck();
  filterBlogCategories();
  initGitHubRepoCards();
  initLightbox();
  initMermaid();
  initScrollIndicator();
  initCodeToggle();
  initKnowledgeChecks();
  initNewBadges();
  initPinnedCarousel();
  initContactHub();
  initSearch();
  initTableOfContents();

  // Initialize Python runners if present
  if ($(".interactive-python").length > 0) {
    initPythonRunners();
  }
});

// Close modal if ESC is pressed
$(document).keyup(function (e) {
  if (e.keyCode === 27) {
    removeModal();
    closeLightbox();
    closeContactHub();
    closeSearch();
    closeShortcuts();
    closeGitHubReposDrawer();
    closeProfileDropdown();
  }
});

$(window).resize(function () {
  $(".header").removeClass("hide-nav"); // Ensure nav will be shown on resize
  $(".header__toggle").removeClass("--open");
  $(".header__links").removeClass("js--open");
  $(".header__links").removeAttr("style"); // If mobile nav was collapsed, make sure it's show on DESK
  $(".header__overlay").remove(); // Remove mobile navigation overlay in case it was opened
  // Reset accordion state on resize
  if (typeof resetMobileAccordion === "function") {
    resetMobileAccordion();
  }
});

/*-------------------------------------------------------------------------*/
/* MOBILE NAVIGATION */
/* -----------------------------------------------------------------------*/

function toggleMobileNav() {
  $(".header__toggle").click(function () {
    if (!$(".header__links").is(".velocity-animating")) {
      if ($(".header__links").hasClass("js--open")) {
        hideMobileNav();
      } else {
        openMobileNav();
      }
    }
  });

  $("body").on("click", function (e) {
    if (e.target.classList.contains("header__overlay")) {
      hideMobileNav();
    }
  });

  // Initialize mobile accordion for Projects dropdown
  initMobileAccordion();
}

/*-------------------------------------------------------------------------*/
/* MOBILE ACCORDION (Projects Dropdown) */
/* -----------------------------------------------------------------------*/

function initMobileAccordion() {
  var $trigger = $(".header__dropdown-trigger");
  var lgBreakpoint = 992; // Matches $lg SCSS variable

  $trigger.on("click", function (e) {
    // Only handle accordion on mobile (below $lg breakpoint)
    if ($(window).width() < lgBreakpoint) {
      e.preventDefault();
      e.stopPropagation();

      var $dropdown = $(this).closest(".header__dropdown");
      var isOpen = $dropdown.hasClass("is-open");

      // Toggle accordion state
      $dropdown.toggleClass("is-open", !isOpen);

      // Update aria-expanded for accessibility
      $(this).attr("aria-expanded", !isOpen);
    }
  });

  // Initialize profile dropdown
  initProfileDropdown();
}

/*-------------------------------------------------------------------------*/
/* PROFILE DROPDOWN (Desktop only - mobile uses hamburger menu) */
/* -----------------------------------------------------------------------*/

function initProfileDropdown() {
  var $profile = $(".header__profile");

  if ($profile.length === 0) return;

  // Initialize profile image fallback (for desktop)
  initProfileImageFallback();

  // Close profile dropdown when clicking contact trigger inside it (desktop)
  $profile.find(".js-contact-trigger").on("click", function () {
    closeProfileDropdown();
  });

  // Close profile dropdown when clicking any link inside it (desktop)
  $profile.find(".header__profile-link").on("click", function () {
    closeProfileDropdown();
  });
}

function initProfileImageFallback() {
  // Handle trigger image
  var $triggerImg = $(".header__profile-trigger .header__profile-img");
  $triggerImg.on("error", function () {
    $(this).closest(".header__profile-trigger").addClass("has-fallback");
  });

  // Handle avatar image in dropdown
  var $avatarImg = $(".header__profile-avatar-wrapper .header__profile-avatar");
  $avatarImg.on("error", function () {
    $(this).closest(".header__profile-avatar-wrapper").addClass("has-fallback");
  });
}

function closeProfileDropdown() {
  var $profile = $(".header__profile");
  var $trigger = $(".js-profile-trigger");
  $profile.removeClass("is-open");
  $trigger.attr("aria-expanded", "false");
}

function resetMobileAccordion() {
  // Reset accordion state when closing menu or resizing
  $(".header__dropdown").removeClass("is-open");
  $(".header__dropdown-trigger").attr("aria-expanded", "false");
}

function openMobileNav() {
  $(".header__links").velocity("slideDown", {
    duration: 300,
    easing: "ease-out",
    display: "block",
    visibility: "visible",
    begin: function () {
      $(".header__toggle").addClass("--open");
      $("body").append("<div class='header__overlay'></div>");
      // Add animating class for staggered reveal
      $(this).addClass("js--animating");
    },
    progress: function () {
      $(".header__overlay").addClass("--open");
    },
    complete: function () {
      $(this).addClass("js--open");
      // Remove animating class after animations complete (500ms buffer)
      var $links = $(this);
      setTimeout(function() {
        $links.removeClass("js--animating");
      }, 500);
    },
  });
}

function hideMobileNav() {
  $(".header__overlay").remove();
  $(".header__links").velocity("slideUp", {
    duration: 300,
    easing: "ease-out",
    display: "none",
    visibility: "hidden",
    begin: function () {
      $(".header__toggle").removeClass("--open");
      // Reset accordion when menu closes
      resetMobileAccordion();
    },
    progress: function () {
      $(".header__overlay").removeClass("--open");
    },
    complete: function () {
      $(this).removeClass("js--open js--animating");
      $(".header__toggle, .header__overlay").removeClass("--open");
    },
  });
}

/*-------------------------------------------------------------------------*/
/* SHOW/SCROLL NAVIGATION */
/* -----------------------------------------------------------------------*/

function ShowHideNav() {
  var previousScroll = 0, // previous scroll position
    $header = $(".header"), // just storing header in a variable
    navHeight = $header.outerHeight(), // nav height
    detachPoint = 576 + 60, // after scroll past this nav will be hidden
    hideShowOffset = 6; // scroll value after which nav will be shown/hidden

  $(window).scroll(function () {
    var wW = 1024;

    // if window width is more than 1024px start show/hide nav
    if ($(window).width() >= wW) {
      if (!$header.hasClass("fixed")) {
        var currentScroll = $(this).scrollTop(),
          scrollDifference = Math.abs(currentScroll - previousScroll);

        // if scrolled past nav
        if (currentScroll > navHeight) {
          // if scrolled past detach point -> show nav
          if (currentScroll > detachPoint) {
            if (!$header.hasClass("fix-nav")) {
              $header.addClass("fix-nav");
            }
          }

          if (scrollDifference >= hideShowOffset) {
            if (currentScroll > previousScroll) {
              // scroll down -> hide nav
              if (!$header.hasClass("hide-nav")) {
                $header.addClass("hide-nav");
              }
            } else {
              // scroll up -> show nav
              if ($header.hasClass("hide-nav")) {
                $($header).removeClass("hide-nav");
              }
            }
          }
        } else {
          // at the top
          if (currentScroll <= 0) {
            $header.removeClass("hide-nav show-nav");
            $header.addClass("top");
          }
        }
      }

      // scrolled to the bottom -> show nav
      if (window.innerHeight + window.scrollY >= document.body.offsetHeight) {
        $header.removeClass("hide-nav");
      }
      previousScroll = currentScroll;
    } else {
      $header.addClass("fix-nav");
    }
  });
}

/*-------------------------------------------------------------------------*/
/* HANDLE MODAL */
/* -----------------------------------------------------------------------*/

function openModal() {
  $("body").css("overflow", "hidden");
  $(".modal, .modal__overlay").show().css("display", "flex");
  $(".modal__inner").velocity({ translateY: 0, opacity: 1 });
  $(".modal__overlay").velocity({ opacity: 1 }, 100);
}

function removeModal() {
  $("body").css({ overflow: "visible" });
  $(".modal, .modal__overlay, .modal__inner").velocity(
    { opacity: 0 },
    function () {
      $(".modal").css({ opacity: 1 });
      $(".modal__inner").css({
        "-webkit-transform": "translateY(200px)",
        "-ms-transform": "translateY(200px)",
        transform: "translateY(200px)",
      });
      $(".modal, .modal__overlay").hide();
      $(".modal__body").empty();
    }
  );
}

$(".js-modal-close").click(function () {
  removeModal();
});

$(".modal__overlay").click(function () {
  removeModal();
});

/*-------------------------------------------------------------------------*/
/* FORM VALIDATION */
/* -----------------------------------------------------------------------*/

function formCheck() {
  $(".js-submit").click(function (e) {
    e.preventDefault();

    var $inputs = $(".form__input input");
    var textarea = $(".form__input textarea");
    var isError = false;

    $(".form__input").removeClass("error");
    $(".error-data").remove();

    for (var i = 0; i < $inputs.length; i++) {
      var input = $inputs[i];
      if (
        $(input).attr("required", true) &&
        !validateRequired($(input).val())
      ) {
        addErrorData($(input), "This field is required");

        isError = true;
      }
      if (
        $(input).attr("required", true) &&
        $(input).attr("type") === "email" &&
        !validateEmail($(input).val())
      ) {
        addErrorData($(input), "Email address is invalid");
        isError = true;
      }
      if (
        $(textarea).attr("required", true) &&
        !validateRequired($(textarea).val())
      ) {
        addErrorData(
          $(textarea),
          "This field is required - is this change getting detected"
        );
        isError = true;
      }
    }
    if (isError === false) {
      $("#contactForm").submit();
    }
  });
}

// Validate if the input is not empty
function validateRequired(value) {
  if (value === "") {
    return false;
  }
  return true;
}

// Validate if the email is using correct format
function validateEmail(value) {
  if (value !== "") {
    return /[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?/i.test(
      value
    );
  }
  return true;
}

// Add error message to the input
function addErrorData(element, error) {
  element.parent().addClass("error");
  element.after("<span class='error-data'>" + error + "</span>");
}

/* Filter navigation */
function filterBlogCategories() {
  $(".filter-click").click(function (e) {
    e.preventDefault();
    var button = $(".filter-click");
    var button_val = button.val();

    return button_val;
  });
}

document.addEventListener("DOMContentLoaded", (event) => {
  // Check if the footnotes div exists
  const footnotesDiv = document.querySelector(".footnotes");
  if (footnotesDiv) {
    // Create the header element
    const header = document.createElement("h2");
    header.className = "footnotes-header";
    header.textContent = "Footnotes";

    // Insert the header before the footnotes content
    footnotesDiv.insertBefore(header, footnotesDiv.firstChild);
  }
});

document.addEventListener("DOMContentLoaded", (event) => {
  document.querySelectorAll("td.f1-score").forEach((cell) => {
    if (cell.textContent !== "N/A") {
      const score = parseFloat(cell.textContent);
      if (score >= 0.8) {
        cell.style.backgroundColor = "#4CAF50"; // Green
      } else if (score >= 0.6) {
        cell.style.backgroundColor = "#FFEB3B"; // Yellow
      } else {
        cell.style.backgroundColor = "#F44336"; // Red
      }
    }
  });
});

document.addEventListener("DOMContentLoaded", (event) => {
  const cells = Array.from(document.querySelectorAll("td.cluster-time")).filter(
    (cell) => cell.textContent !== "N/A"
  );
  const scores = cells.map((cell) => parseFloat(cell.textContent));
  // Apply logarithm + 1 to avoid log(0)
  const logScores = scores.map((score) => Math.log1p(score));
  const minLogScore = Math.min(...logScores);
  const maxLogScore = Math.max(...logScores);

  function interpolateColor(logScore, minLogScore, maxLogScore) {
    // Adjust the color interpolation to work on a logarithmic scale
    const fraction = (logScore - minLogScore) / (maxLogScore - minLogScore);
    const r = Math.round(255 * fraction); // Higher logScore gets more red
    const g = Math.round(255 * (1 - fraction)); // Lower logScore gets more green
    const b = 0; // Blue remains constant, adjust if desired
    return `rgb(${r},${g},${b})`;
  }

  cells.forEach((cell) => {
    const score = parseFloat(cell.textContent);
    const logScore = Math.log1p(score); // Apply logarithm + 1 to score for color calculation
    const color = interpolateColor(logScore, minLogScore, maxLogScore);
    cell.style.backgroundColor = color;
  });
});

/*-------------------------------------------------------------------------*/
/* AJAX FORM SUBMIT
/* Formspree now only supports AJAX for Gold Users
/* https://github.com/formspree/formspree/pull/173
/* Uncomment if you want to use AJAX Form submission and you're a gold user
/* -----------------------------------------------------------------------*/

// $( "#contactForm" ).submit( function( e ) {

//     e.preventDefault();

//     var $btn = $( ".js-submit" ),
//         $inputs = $( ".form__input input" ),
//         $textarea = $( ".form__input textarea" ),
//         $name = $( "input#name" ).val(),
//         $url = $( "#contactForm" ).attr( "action" );

//     $.ajax( {

//         url: $url,
//         method: "POST",
//         data: $( this ).serialize(),
//         dataType: "json",

//         beforeSend: function() {
//             $btn.prop( "disabled", true );
//             $btn.text( "Sending..." );
//         },
//         // eslint-disable-next-line no-unused-vars
//         success: function( data ) {
//             $inputs.val( "" );
//             $textarea.val( "" );
//             $btn.prop( "disabled", false );
//             $btn.text( "Send" );
//             openModal();
//             $( ".modal__body" ).append(
//               "<h1>Thanks " +
//               $name +
//               "!</h1><p>Your message was successfully sent! Will get back to you soon.</p>"
//             );

//         },
//         error: function( err ) {
//             $( ".modal, .modal__overlay" ).addClass( "--show" );
//             $( ".modal__body" ).append(
//               "<h1>Aww snap!</h1><p>Something went wrong, please try again. Error message: </p>" +
//               err
//             );
//         }
//     } );
// } );

/*-------------------------------------------------------------------------*/
/* GITHUB REPOSITORY CARDS */
/* -----------------------------------------------------------------------*/

function initGitHubRepoCards() {
  // Language colors mapping
  const languageColors = {
    JavaScript: "#f1e05a",
    TypeScript: "#2b7489",
    Python: "#3572A5",
    Java: "#b07219",
    "C++": "#f34b7d",
    C: "#555555",
    Go: "#00ADD8",
    Rust: "#dea584",
    Ruby: "#701516",
    PHP: "#4F5D95",
    Swift: "#ffac45",
    Kotlin: "#A97BFF",
    Scala: "#c22d40",
    Shell: "#89e051",
    HTML: "#e34c26",
    CSS: "#563d7c",
    Vue: "#4fc08d",
    React: "#61dafb",
    Angular: "#dd0031",
    Svelte: "#ff3e00",
    "Jupyter Notebook": "#DA5B0B",
  };

  $(".github-repo-card").each(function () {
    const $card = $(this);
    const repo = $card.data("repo");

    if (!repo) {
      console.warn("GitHub repo card missing data-repo attribute");
      return;
    }

    const $loading = $card.find(".github-repo-loading");
    const $content = $card.find(".github-repo-content");
    const $error = $card.find(".github-repo-error");

    fetch(`https://api.github.com/repos/${repo}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (data.message) {
          console.error("GitHub API error message:", data.message);
          $loading.hide();
          $error.show();
          $error.find("p").text(`Error: ${data.message}`);
          return;
        }

        // Update repository name and link
        const $nameLink = $content.find(".github-repo-name a");
        $nameLink.text(data.name);
        $nameLink.attr("href", data.html_url);

        // Update author information
        const $authorLink = $content.find(".github-repo-author a");
        $authorLink.text(data.owner.login);
        $authorLink.attr("href", data.owner.html_url);

        // Update description
        const $description = $content.find(".github-repo-description");
        $description.text(data.description || "No description available");

        // Update language
        const $languageSpan = $content.find(".github-repo-language");
        if (data.language) {
          const $colorDot = $languageSpan.find(".language-color");
          const $langName = $languageSpan.find(".language-name");
          $colorDot.css(
            "background-color",
            languageColors[data.language] || "#ccc"
          );
          $langName.text(data.language);
          $languageSpan.show();
        } else {
          $languageSpan.hide();
        }

        // Update stars and forks
        $content
          .find(".stars-count")
          .text((data.stargazers_count || 0).toLocaleString());
        $content
          .find(".forks-count")
          .text((data.forks_count || 0).toLocaleString());

        // Update topics
        const $topicsContainer = $content.find(".github-repo-topics");
        if (data.topics && data.topics.length > 0) {
          const topicsHtml = data.topics
            .map((topic) => `<span class="github-repo-topic">${topic}</span>`)
            .join("");
          $topicsContainer.html(topicsHtml);
          $topicsContainer.show();
        }

        // Show content, hide loading
        $loading.hide();
        $content.show();
      })
      .catch((err) => {
        console.error("Error fetching GitHub repository:", err);
        $loading.hide();
        $error.show();
        $error.find("p").text(`Error: ${err.message}`);
      });
  });
}

/*-------------------------------------------------------------------------*/
/* LIGHTBOX FUNCTIONALITY */
/* -----------------------------------------------------------------------*/

function initLightbox() {
  // Create lightbox overlay HTML
  const lightboxHTML = `
    <div class="lightbox-overlay" id="lightbox-overlay">
      <div class="lightbox-content">
        <button class="lightbox-close" id="lightbox-close">&times;</button>
        <div class="lightbox-image-container">
          <img id="lightbox-image" src="" alt="">
        </div>
        <div class="lightbox-caption" id="lightbox-caption"></div>
      </div>
    </div>
  `;

  // Add lightbox to body
  $("body").append(lightboxHTML);

  // Click handlers for lightbox images
  $(document).on("click", ".lightbox-image", function (e) {
    e.preventDefault();
    openLightbox(this);
  });

  // Click handler for hero images
  $(document).on("click", ".hero--clickable", function (e) {
    // Don't trigger if clicking on a link inside the hero
    if ($(e.target).closest("a").length) return;

    const src = $(this).data("lightbox-src");
    const caption = $(this).data("lightbox-caption") || "";

    if (src) {
      openHeroLightbox(src, caption);
    }
  });

  // Keyboard support for hero lightbox
  $(document).on("keydown", ".hero--clickable", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      $(this).trigger("click");
    }
  });

  // Close lightbox handlers
  $("#lightbox-close").click(closeLightbox);
  $("#lightbox-overlay").click(function (e) {
    if (e.target === this) {
      closeLightbox();
    }
  });
}

function openLightbox(imageElement) {
  const $img = $(imageElement);
  const src = $img.attr("src");
  const alt = $img.attr("alt") || "";

  // Find caption from next sibling with .image-caption class
  let caption = "";
  const $nextElement = $img.parent().next(".image-caption");
  if ($nextElement.length) {
    caption = $nextElement.text();
  }

  $("#lightbox-image").attr("src", src).attr("alt", alt);
  $("#lightbox-caption").text(caption);

  $("body").css("overflow", "hidden");
  $(".lightbox-overlay").css("display", "flex").hide().fadeIn(300);
}

function closeLightbox() {
  $(".lightbox-overlay").fadeOut(300, function () {
    $("body").css("overflow", "visible");
    $("#lightbox-image").attr("src", "");
    $("#lightbox-caption").text("");
  });
}

function openHeroLightbox(src, caption) {
  $("#lightbox-image").attr("src", src).attr("alt", caption);
  $("#lightbox-caption").text(caption);

  $("body").css("overflow", "hidden");
  $(".lightbox-overlay").css("display", "flex").hide().fadeIn(300);
}

/*-------------------------------------------------------------------------*/
/* MERMAID DIAGRAM INITIALIZATION */
/* -----------------------------------------------------------------------*/

function initMermaid() {
  // Check if there are any mermaid code blocks or elements on the page
  const hasMermaidContent =
    $("code.language-mermaid").length > 0 || $(".mermaid").length > 0;

  if (!hasMermaidContent) {
    // No Mermaid content found, skip initialization to avoid unnecessary polling
    return;
  }

  // Wait for Mermaid library to be available
  function waitForMermaid() {
    if (typeof window.mermaid !== "undefined") {
      console.log("Mermaid library loaded, initializing...");
      setupMermaid();
    } else {
      console.log("Waiting for Mermaid library...");
      setTimeout(waitForMermaid, 100);
    }
  }

  waitForMermaid();
}

function setupMermaid() {
  // Convert Jekyll syntax-highlighted code blocks to Mermaid divs
  $("code.language-mermaid").each(function () {
    const $code = $(this);
    const $pre = $code.closest("pre");
    const $div = $("<div></div>");

    $div.addClass("mermaid");
    const content = $code.text().trim();
    $div.text(content);

    console.log("Converting mermaid code block:", content);
    $pre.replaceWith($div);
  });

  // Initialize Mermaid - always use dark theme based on your screenshot
  const isDark =
    window.matchMedia("(prefers-color-scheme: dark)").matches ||
    document.documentElement.classList.contains("dark") ||
    document.body.classList.contains("dark");
  console.log("Theme detected:", "dark (forced for mindmaps)");

  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: "dark",
    themeVariables: {
      primaryColor: "#ff6b6b",
      primaryTextColor: "#fff",
      primaryBorderColor: "#444",
      lineColor: "#888",
      // Mindmap specific colors
      mindmapLabelBackgroundColor: "transparent",
      mindmapNodeBackgroundColor: "#374151",
      mindmapNodeTextColor: "#fff",
    },
  });

  // Find and render Mermaid diagrams
  const $mermaidElements = $(".mermaid");
  console.log("Found mermaid elements:", $mermaidElements.length);

  if ($mermaidElements.length > 0) {
    window.mermaid
      .run({
        querySelector: ".mermaid",
      })
      .then(() => {
        console.log("Mermaid diagrams rendered successfully");

        // Add pan/zoom controls to each rendered diagram
        $(".mermaid").each(function () {
          addPanZoomToMermaid(this);
        });
      })
      .catch((error) => {
        console.error("Error rendering Mermaid diagrams:", error);
      });
  }
}

function addPanZoomToMermaid(container) {
  const $container = $(container);
  const svg = container.querySelector("svg");

  if (!svg || $container.closest(".mermaid-wrapper").length > 0) {
    return; // Already wrapped or no SVG
  }

  // Add pan/zoom to all Mermaid diagrams (including mindmaps)
  console.log("Adding pan/zoom to diagram with SVG ID:", svg.id);

  // Wait for panzoom library to be available
  if (typeof window.panzoom === "undefined") {
    console.log("Panzoom library not yet loaded, will retry...");
    setTimeout(() => addPanZoomToMermaid(container), 100);
    return;
  }

  console.log("Adding pan/zoom controls to Mermaid diagram");

  // Create wrapper
  const $wrap = $('<div class="mermaid-wrapper"></div>');
  $wrap.css({
    width: "100%",
    height: "min(75vh, 600px)",
    overflow: "hidden",
    position: "relative",
    border: "1px solid #444",
    borderRadius: "8px",
    background: "#1a1a1a",
    margin: "1.5rem 0",
  });

  $container.before($wrap);
  $wrap.append($container);

  // Initialize panzoom
  const pz = window.panzoom(svg, {
    maxZoom: 4,
    minZoom: 0.5,
    smoothScroll: false,
  });

  // Create control buttons
  const $controls = $('<div class="mermaid-controls"></div>');
  $controls.css({
    position: "absolute",
    top: ".75rem",
    right: ".75rem",
    display: "flex",
    gap: ".25rem",
    zIndex: 10,
  });

  const buttons = [
    {
      text: "↺",
      action: () => {
        pz.moveTo(0, 0);
        pz.zoomAbs(0, 0, 1);
      },
    },
  ];

  buttons.forEach((btn) => {
    const $button = $('<button class="mermaid-btn"></button>');
    $button.text(btn.text);
    $button.attr("type", "button");
    $button.click(btn.action);
    $controls.append($button);
  });

  $wrap.append($controls);

  console.log("Pan/zoom controls added successfully");
}

// Initialize scroll indicator for pinned/year sections
function initScrollIndicator() {
  const $indicator = $(".scroll-indicator");
  if ($indicator.length === 0) return;

  const $items = $indicator.find(".scroll-indicator__item");
  if ($items.length === 0) return;

  // Find all post cards and organize by section
  const $allPosts = $(".post-card");
  if ($allPosts.length === 0) return;

  const sections = [];

  // Check for pinned posts
  const $pinnedPosts = $allPosts.filter('[data-pinned="true"]');
  if ($pinnedPosts.length > 0) {
    const $indicatorItem = $items.filter('[data-section="pinned"]');
    if ($indicatorItem.length > 0) {
      sections.push({
        name: "pinned",
        $item: $indicatorItem,
        $firstPost: $pinnedPosts.first(),
      });
    }
  }

  // Group regular posts by year and find first post of each year
  const yearMap = new Map();
  $allPosts.not('[data-pinned="true"]').each(function () {
    const year = $(this).data("year");
    if (year && !yearMap.has(year)) {
      yearMap.set(year, $(this));
    }
  });

  // Add year sections in order
  yearMap.forEach((firstPost, year) => {
    const $indicatorItem = $items.filter('[data-section="year-' + year + '"]');
    if ($indicatorItem.length > 0) {
      sections.push({
        name: "year-" + year,
        $item: $indicatorItem,
        $firstPost: firstPost,
      });
    }
  });

  if (sections.length === 0) return;

  // Update active section based on scroll position
  function updateActiveSection() {
    const scrollTop = $(window).scrollTop();
    const windowHeight = $(window).height();
    const scrollMid = scrollTop + windowHeight / 3;

    // Check if we're past the hero section
    const hero = $(".hero");
    const heroHeight =
      hero.length > 0 ? hero.offset().top + hero.outerHeight() : 0;

    if (scrollTop + 200 > heroHeight) {
      $indicator.addClass("visible");
    } else {
      $indicator.removeClass("visible");
    }

    let activeSection = sections[0];

    // Find which section we're in
    for (let i = 0; i < sections.length; i++) {
      const section = sections[i];
      if (section.$firstPost.length > 0) {
        const postTop = section.$firstPost.offset().top;
        if (scrollMid >= postTop) {
          activeSection = section;
        }
      }
    }

    // Update active state
    $items.removeClass("active");
    if (activeSection) {
      activeSection.$item.addClass("active");
    }
  }

  // Throttle scroll events
  let scrollTimeout;
  $(window).on("scroll", function () {
    if (scrollTimeout) clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(updateActiveSection, 50);
  });

  // Initial update
  updateActiveSection();

  // Click to scroll to section
  $items.on("click", function (e) {
    e.preventDefault();
    const sectionName = $(this).data("section");
    const section = sections.find((s) => s.name === sectionName);

    if (section && section.$firstPost.length > 0) {
      $("html, body").animate(
        {
          scrollTop: section.$firstPost.offset().top - 100,
        },
        600
      );
    }
  });
}

/*-------------------------------------------------------------------------*/
/* TABLE OF CONTENTS */
/* -----------------------------------------------------------------------*/

function initTableOfContents() {
  var $toc = $('.toc');
  if ($toc.length === 0) return;

  var $toggle = $toc.find('.toc__toggle');
  var $nav = $toc.find('.toc__nav');

  // Generate TOC from headers (h1, h2, h3, h4) - skip "Table of Contents" header
  var $headers = $('.post-content').find('h1, h2, h3, h4').filter(function() {
    var text = $(this).text().trim();
    // For comparison, strip emojis/special chars but check original text isn't empty
    var textForComparison = text.toLowerCase().replace(/[^\w\s]/g, '').trim();
    return textForComparison !== 'table of contents' && text.length > 0;
  });

  if ($headers.length === 0) {
    $toc.hide();
    return;
  }

  var $list = $('<ul class="toc__list"></ul>');
  $headers.each(function () {
    var $h = $(this);
    // Get the full text content including emojis
    var text = $h.text().trim();
    var level = this.tagName.toLowerCase();
    var id = $h.attr('id');
    if (!id) {
      // Generate ID from text, removing emojis and special chars for valid ID
      id = 'toc-' + text.toLowerCase()
        .replace(/[\u{1F300}-\u{1FAF8}]/gu, '') // Remove emojis from ID
        .replace(/[^\w\s-]/g, '')
        .trim()
        .replace(/\s+/g, '-')
        .substring(0, 50);
      $h.attr('id', id);
    }

    var $item = $('<li class="toc__item toc__item--' + level + '"></li>');
    var $link = $('<a class="toc__link" href="#' + id + '"></a>');
    $link.text(text); // Preserves emojis
    $item.append($link);
    $list.append($item);
  });
  $nav.append($list);

  // Toggle functionality - starts collapsed on all devices
  $toggle.on('click', function () {
    var expanded = $(this).attr('aria-expanded') === 'true';
    $(this).attr('aria-expanded', !expanded);
    $nav.toggleClass('is-open');
  });

  // Scroll spy
  var $links = $toc.find('.toc__link');
  var scrollTimer;
  $(window).on('scroll.toc', function () {
    clearTimeout(scrollTimer);
    scrollTimer = setTimeout(function () {
      var scrollPos = $(window).scrollTop() + 100;
      var activeId = null;
      $headers.each(function () {
        if (scrollPos >= $(this).offset().top) {
          activeId = $(this).attr('id');
        }
      });
      $links.removeClass('is-active');
      if (activeId) {
        $links.filter('[href="#' + activeId + '"]').addClass('is-active');
      }
    }, 50);
  });

  // Smooth scroll on click
  $links.on('click', function (e) {
    e.preventDefault();
    var $target = $($(this).attr('href'));
    if ($target.length) {
      $('html, body').animate({ scrollTop: $target.offset().top - 80 }, 400);
      // Close on mobile after click
      if ($(window).width() < 992) {
        $toggle.attr('aria-expanded', 'false');
        $nav.removeClass('is-open');
      }
    }
  });
}

/*-------------------------------------------------------------------------*/
/* CODE TOGGLE FUNCTIONALITY */
/* -----------------------------------------------------------------------*/

function initCodeToggle() {
  $(".code-toggle__tab").on("click", function () {
    const $tab = $(this);
    const $toggle = $tab.closest(".code-toggle");
    const targetTab = $tab.data("tab");

    // Update tab active states
    $toggle.find(".code-toggle__tab").removeClass("code-toggle__tab--active");
    $tab.addClass("code-toggle__tab--active");

    // Update pane active states
    $toggle.find(".code-toggle__pane").removeClass("code-toggle__pane--active");
    $toggle
      .find(`.code-toggle__pane[data-pane="${targetTab}"]`)
      .addClass("code-toggle__pane--active");
  });
}

/*-------------------------------------------------------------------------*/
/* KNOWLEDGE CHECK FUNCTIONALITY */
/* -----------------------------------------------------------------------*/

function initKnowledgeChecks() {
  const $quizzes = $(".markdown-quiz");

  if ($quizzes.length === 0) return;

  $quizzes.each(function (index) {
    const $quiz = $(this);
    let quizId = $quiz.data("quiz-id");

    if (!quizId) {
      quizId = `markdown-quiz-${index + 1}`;
      $quiz.attr("data-quiz-id", quizId);
    }

    const $options = $quiz.find(".markdown-quiz__option");
    const $feedback = $quiz.find(".markdown-quiz__feedback");
    const $submit = $quiz.find(".markdown-quiz__submit");
    const $reset = $quiz.find(".markdown-quiz__reset");
    const $explanation = $quiz.find(".markdown-quiz__explanation");

    if ($options.length === 0 || $submit.length === 0) return;

    function parseBoolean(value) {
      if (typeof value === "string") {
        return value.toLowerCase() === "true";
      }
      return Boolean(value);
    }

    function clearState() {
      $quiz.removeClass("is-correct is-incorrect has-feedback");
      $options.removeClass("is-selected is-answer is-wrong");
      $feedback.text("");
      if ($explanation.length) {
        $explanation.attr("hidden", true);
      }
    }

    clearState();

    $options.each(function () {
      const $option = $(this);
      const $input = $option.find('input[type="radio"]');

      if (quizId && $input.length && !$input.attr("name")) {
        $input.attr("name", quizId);
      }

      $input.on("change", function () {
        $options.removeClass("is-selected");
        if ($input.is(":checked")) {
          $option.addClass("is-selected");
        }
        $quiz.removeClass("is-correct is-incorrect has-feedback");
        $options.removeClass("is-answer is-wrong");
        $feedback.text("");
        if ($explanation.length) {
          $explanation.attr("hidden", true);
        }
      });

      $option.on("click", function (event) {
        const tag = event.target.tagName.toLowerCase();
        if (tag !== "input" && tag !== "label") {
          $input.prop("checked", true).trigger("change");
        }
      });
    });

    $submit.on("click", function (event) {
      event.preventDefault();

      const $selected = $options.filter(".is-selected");
      if ($selected.length === 0) {
        $feedback.text("Select an option before checking the answer.");
        $quiz.addClass("has-feedback");
        return;
      }

      const isCorrect = parseBoolean($selected.data("correct"));
      const customFeedback = $selected.data("feedback");
      const $correctOption = $options.filter(function () {
        return parseBoolean($(this).data("correct"));
      });

      $options.removeClass("is-answer is-wrong");

      if (isCorrect) {
        $quiz.addClass("is-correct").removeClass("is-incorrect");
        $selected.addClass("is-answer");
        $feedback.text(customFeedback || "Great job! That's correct.");
      } else {
        $quiz.addClass("is-incorrect").removeClass("is-correct");
        $selected.addClass("is-wrong");
        if ($correctOption.length > 0) {
          $correctOption.first().addClass("is-answer");
        }
        $feedback.text(
          customFeedback || "Not quite. Review the explanation and try again."
        );
      }

      if ($explanation.length) {
        $explanation.attr("hidden", false);
      }

      $quiz.addClass("has-feedback");
    });

    if ($reset.length) {
      $reset.on("click", function (event) {
        event.preventDefault();
        clearState();
        $quiz.find('input[type="radio"]').prop("checked", false);
      });
    }
  });
}

/*-------------------------------------------------------------------------*/
/* NEW BADGES FUNCTIONALITY */
/* -----------------------------------------------------------------------*/

function initNewBadges() {
  const twoWeeksAgo = new Date();
  twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);

  $(".post-card").each(function () {
    const $card = $(this);
    const dateStr = $card.data("date");

    if (!dateStr) return;

    const postDate = new Date(dateStr);

    if (postDate >= twoWeeksAgo) {
      $card.addClass("new");
      const $placeholder = $card.find(".post-card__new-badge-placeholder");
      if ($placeholder.length > 0) {
        $placeholder.replaceWith(
          '<span class="label post-card__new-badge">✨ NEW</span>'
        );
      }
    }
  });
}

/*-------------------------------------------------------------------------*/
/* PINNED CAROUSEL FUNCTIONALITY */
/* -----------------------------------------------------------------------*/

function initPinnedCarousel() {
  const $track = $("#pinned-carousel-track");
  if ($track.length === 0) return;

  const trackEl = $track[0];
  const $left  = $("#carousel-arrow-left");
  const $right = $("#carousel-arrow-right");

  // Prevent duplicate bindings if this is re-initialized
  $left.off("click.pinned");
  $right.off("click.pinned");
  $track.off("scroll.pinned");

  // Helper to calculate max scroll distance
  const maxScroll = () => Math.max(0, trackEl.scrollWidth - trackEl.clientWidth);

  // Update UI state (arrow buttons, fade indicators)
  let ticking = false;
  function updateUI() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const sl = $track.scrollLeft();
      const ms = maxScroll();

      $track.toggleClass("has-scroll-left",  sl > 10);
      $track.toggleClass("has-scroll-right", sl < ms - 10);
      $left.prop("disabled",  sl <= 10);
      $right.prop("disabled", sl >= ms - 10);

      ticking = false;
    });
  }

  // Arrow button click handler - scroll by 2 cards
  function scrollByCards(dir = 1) {
    const $card = $track.find(".pinned-section__card").first();
    const cardW = $card.length ? Math.round($card.outerWidth(true)) : Math.round(trackEl.clientWidth * 0.8);
    const delta = dir < 0 ? -(cardW * 2) : (cardW * 2);

    if (trackEl.scrollBy) {
      trackEl.scrollBy({ left: delta, behavior: "smooth" });
    } else {
      const target = Math.max(0, Math.min(maxScroll(), $track.scrollLeft() + delta));
      $track.animate({ scrollLeft: target }, 300);
    }
  }

  // Bind arrow button clicks
  $left.on("click.pinned",  (e) => { e.preventDefault(); scrollByCards(-1); });
  $right.on("click.pinned", (e) => { e.preventDefault(); scrollByCards(+1); });

  // Update UI on scroll and resize
  $track.on("scroll.pinned", updateUI);
  $(window).on("resize.pinned", updateUI);

  // Initial state
  updateUI();
}

/*-------------------------------------------------------------------------*/
/* CONTACT HUB MODAL */
/* -----------------------------------------------------------------------*/

function initContactHub() {
  const $hub = $(".contact-hub");
  const $trigger = $(".js-contact-trigger");
  const $close = $(".js-contact-hub-close");

  if ($hub.length === 0) return;

  // Open contact hub
  $trigger.on("click", function (e) {
    e.preventDefault();
    openContactHub();
  });

  // Close contact hub
  $close.on("click", function (e) {
    e.preventDefault();
    closeContactHub();
  });

  // Form validation for contact hub
  initContactHubForm();
}

function openContactHub() {
  const $hub = $(".contact-hub");
  if ($hub.length === 0) return;

  // Prevent body scroll
  $("body").css("overflow", "hidden");

  // Show and animate
  $hub.addClass("is-active");

  // Focus the first input after animation
  setTimeout(function () {
    $hub.find("input, textarea").first().focus();
  }, 600);

  // Set up focus trapping
  trapFocus($hub[0]);
}

function closeContactHub() {
  const $hub = $(".contact-hub");
  if (!$hub.hasClass("is-active")) return;

  // Add closing class for exit animation
  $hub.addClass("is-closing");

  // Wait for animation then hide
  setTimeout(function () {
    $hub.removeClass("is-active is-closing");
    $("body").css("overflow", "");

    // Return focus to trigger button
    $(".js-contact-trigger").first().focus();
  }, 300);
}

function initContactHubForm() {
  const $form = $("#contactHubForm");
  if ($form.length === 0) return;

  $form.on("submit", function (e) {
    const $fields = $form.find(".contact-hub__field");
    let hasError = false;

    // Clear previous errors
    $fields.removeClass("is-error");

    // Validate required fields
    $fields.each(function () {
      const $field = $(this);
      const $input = $field.find("input, textarea");

      if ($input.prop("required") && !$input.val().trim()) {
        $field.addClass("is-error");
        hasError = true;
      }

      // Validate email format
      if ($input.attr("type") === "email" && $input.val().trim()) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test($input.val().trim())) {
          $field.addClass("is-error");
          hasError = true;
        }
      }
    });

    if (hasError) {
      e.preventDefault();
      // Focus first error field
      $fields.filter(".is-error").first().find("input, textarea").focus();
    }
  });

  // Clear error on input
  $form.find("input, textarea").on("input", function () {
    $(this).closest(".contact-hub__field").removeClass("is-error");
  });
}

// Focus trap utility for accessibility
function trapFocus(element) {
  const focusableElements = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const firstFocusable = focusableElements[0];
  const lastFocusable = focusableElements[focusableElements.length - 1];

  function handleTabKey(e) {
    if (e.key !== "Tab") return;

    if (e.shiftKey) {
      if (document.activeElement === firstFocusable) {
        e.preventDefault();
        lastFocusable.focus();
      }
    } else {
      if (document.activeElement === lastFocusable) {
        e.preventDefault();
        firstFocusable.focus();
      }
    }
  }

  // Add listener
  element.addEventListener("keydown", handleTabKey);

  // Store cleanup function
  element._trapFocusCleanup = function () {
    element.removeEventListener("keydown", handleTabKey);
  };
}

/*-------------------------------------------------------------------------*/
/* SEARCH MODAL (Enhanced with Vector Search)                              */
/*-------------------------------------------------------------------------*/

// Mobile: Default to keyword-only search (avoids loading ~23MB semantic model)
var isMobileSearch = window.matchMedia("(max-width: 576px)").matches;
var searchMode = isMobileSearch ? "keyword" : "hybrid"; // 'keyword', 'semantic', or 'hybrid'
var vectorSearchDebounceTimer = null;
var hybridSearchDebounceTimer = null;

function initSearch() {
  var $modal = $(".search-modal");
  var $trigger = $(".js-search-trigger");
  var $close = $(".js-search-close");
  var $modeToggle = $(".search-mode-toggle__btn");

  if ($modal.length === 0) return;

  // Click trigger to open
  $trigger.on("click", function (e) {
    e.preventDefault();
    openSearch();
  });

  // Click close button or overlay to close
  $close.on("click", function (e) {
    e.preventDefault();
    closeSearch();
  });

  // Mode toggle buttons
  $modeToggle.on("click", function (e) {
    e.preventDefault();
    var mode = $(this).data("mode");
    switchSearchMode(mode);
  });

  // Vector search input handler
  var $vectorInput = $(".vector-search__input");
  $vectorInput.on("input", function () {
    clearTimeout(vectorSearchDebounceTimer);
    var query = $(this).val();

    vectorSearchDebounceTimer = setTimeout(function () {
      performVectorSearch(query);
    }, 300); // Debounce 300ms
  });

  // Handle Enter key in vector search input
  $vectorInput.on("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      clearTimeout(vectorSearchDebounceTimer);
      performVectorSearch($(this).val());
    }
  });

  // Hybrid search input handler
  var $hybridInput = $(".hybrid-search__input");
  $hybridInput.on("input", function () {
    clearTimeout(hybridSearchDebounceTimer);
    var query = $(this).val();

    hybridSearchDebounceTimer = setTimeout(function () {
      performHybridSearch(query);
    }, 300);
  });

  // Handle Enter key in hybrid search input
  $hybridInput.on("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      clearTimeout(hybridSearchDebounceTimer);
      performHybridSearch($(this).val());
    }
  });

  // Keyboard shortcut: Cmd+K (Mac) / Ctrl+K (Win/Linux)
  $(document).on("keydown", function (e) {
    // Check for Cmd+K or Ctrl+K
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      if ($modal.hasClass("is-active")) {
        closeSearch();
      } else {
        openSearch();
      }
    }
  });
}

function switchSearchMode(mode, forceInit) {
  if (mode === searchMode && !forceInit) return;

  searchMode = mode;

  // Update toggle buttons
  $(".search-mode-toggle__btn").each(function () {
    var $btn = $(this);
    var isActive = $btn.data("mode") === mode;
    $btn.toggleClass("search-mode-toggle__btn--active", isActive);
    $btn.attr("aria-checked", isActive ? "true" : "false");
  });

  // Switch visible content
  if (mode === "keyword") {
    $("#pagefind-container").show();
    $("#vector-search-container").hide();
    $("#hybrid-search-container").hide();

    // Focus Pagefind input
    setTimeout(function () {
      $(".pagefind-ui__search-input").focus();
    }, 100);
  } else if (mode === "semantic") {
    $("#pagefind-container").hide();
    $("#vector-search-container").show();
    $("#hybrid-search-container").hide();

    // Initialize vector search if needed
    initVectorSearch();
  } else if (mode === "hybrid") {
    $("#pagefind-container").hide();
    $("#vector-search-container").hide();
    $("#hybrid-search-container").show();

    // Initialize hybrid search (needs vector search model)
    initHybridSearch();
  }
}

function initVectorSearch() {
  if (window.VectorSearch.isAvailable()) {
    // Already initialized, focus input
    $(".vector-search__loading").hide();
    $(".vector-search__input-wrapper").show();
    $(".vector-search__input").focus();
    return;
  }

  if (window.VectorSearch.isLoading()) {
    return; // Already loading
  }

  // Show loading state
  $(".vector-search__loading").show();
  $(".vector-search__input-wrapper").hide();
  $(".vector-search__results").empty();
  $(".vector-search__empty").hide();

  // Initialize with progress callback
  window.VectorSearch.initialize(function (info) {
    $(".vector-search__loading-message").text(info.message);
    $(".vector-search__loading-progress").css("width", info.progress + "%");

    if (info.stage === "ready") {
      $(".vector-search__loading").hide();
      $(".vector-search__input-wrapper").show();
      $(".vector-search__input").focus();
    } else if (info.stage === "error") {
      $(".vector-search__loading-message").text("Failed to load: " + info.message);
      $(".vector-search__loading-spinner").hide();
    }
  });
}

function performVectorSearch(query) {
  if (!window.VectorSearch.isAvailable()) return;

  var $results = $(".vector-search__results");
  var $empty = $(".vector-search__empty");

  if (!query || query.trim().length < 2) {
    $results.empty();
    $empty.hide();
    return;
  }

  window.VectorSearch.search(query, 10, 0.3)
    .then(function (results) {
      renderVectorResults(results);
    })
    .catch(function (err) {
      console.error("Vector search error:", err);
      $results.empty();
      $empty.show().find("p").text("Search error: " + err.message);
    });
}

function renderVectorResults(results) {
  var $container = $(".vector-search__results");
  var $empty = $(".vector-search__empty");

  $container.empty();

  if (results.length === 0) {
    $empty.show();
    return;
  }

  $empty.hide();

  results.forEach(function (result, index) {
    var chunk = result.chunk;
    var similarity = (result.similarity * 100).toFixed(0);

    // Main result card with stagger animation delay
    var $result = $('<a class="vector-search__result" href="' + chunk.url + '"></a>');
    $result.css("animation-delay", (index * 0.04) + "s");

    // Header row: title + score
    var $header = $('<div class="vector-search__result-header"></div>');

    var $title = $('<span class="vector-search__result-title"></span>').text(
      chunk.title
    );

    // Score with visual indicator
    var $score = $('<div class="vector-search__result-score"></div>');
    var scoreClass = similarity >= 70 ? "high" : similarity >= 50 ? "medium" : "low";
    $score.addClass("vector-search__result-score--" + scoreClass);
    $score.html('<span class="score-value">' + similarity + '</span><span class="score-percent">%</span>');

    $header.append($title, $score);

    // Meta row: section tag (if different from title)
    var $meta = $('<div class="vector-search__result-meta"></div>');
    if (chunk.section && chunk.section !== chunk.title) {
      var $section = $('<span class="vector-search__result-section"></span>');
      $section.html('<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h7"/></svg>' + chunk.section);
      $meta.append($section);
    }

    // Excerpt with truncation
    var excerptText = chunk.text.length > 150 ? chunk.text.substring(0, 150) + "..." : chunk.text;
    var $excerpt = $('<p class="vector-search__result-excerpt"></p>').text(excerptText);

    // Categories footer
    var $footer = $('<div class="vector-search__result-footer"></div>');
    if (chunk.categories && chunk.categories.length > 0) {
      var $categories = $('<div class="vector-search__result-categories"></div>');
      chunk.categories.slice(0, 3).forEach(function (cat) {
        $categories.append(
          $('<span class="vector-search__result-category"></span>').text(cat)
        );
      });
      $footer.append($categories);
    }

    $result.append($header, $meta, $excerpt, $footer);
    $container.append($result);
  });
}

function initHybridSearch() {
  var $status = $(".hybrid-search__status");
  var $inputWrapper = $(".hybrid-search__input").closest(".vector-search__input-wrapper");
  var $loading = $(".hybrid-search__loading");

  if (window.VectorSearch.isAvailable()) {
    // Model ready - hide loading/status, show input
    $loading.hide();
    $inputWrapper.show();
    $status.hide();
    $(".hybrid-search__input").focus();
    return;
  }

  // Always show input immediately (allow keyword-only search while model loads)
  $loading.hide();
  $inputWrapper.show();
  $(".hybrid-search__results").empty();
  $(".hybrid-search__empty").hide();
  $(".hybrid-search__input").focus();

  // Show status badge indicating model is loading
  $status.html('<span class="hybrid-search__status-spinner"></span> Loading semantic model...').show();

  if (window.VectorSearch.isLoading()) {
    return; // Already loading, status badge is showing
  }

  // Initialize with progress callback
  window.VectorSearch.initialize(function (info) {
    if (info.stage === "ready") {
      // Model ready - hide status badge
      $status.hide();
    } else if (info.stage === "error") {
      // Model failed - show keyword-only status
      $status.html('Keyword only (semantic unavailable)').show();
    }
  });
}

function loadPagefindAPI() {
  if (window.pagefindAPI) {
    return Promise.resolve(window.pagefindAPI);
  }

  // Use dynamic import for ES module
  return import("/pagefind/pagefind.js")
    .then(function (pagefind) {
      return pagefind.init().then(function () {
        window.pagefindAPI = pagefind;
        return pagefind;
      });
    })
    .catch(function (err) {
      console.error("[HybridSearch] Failed to load Pagefind:", err);
      return null;
    });
}

function performHybridSearch(query) {
  var $results = $(".hybrid-search__results");
  var $empty = $(".hybrid-search__empty");

  if (!query || query.trim().length < 2) {
    $results.empty();
    $empty.hide();
    return;
  }

  // Run both searches in parallel
  var keywordPromise = loadPagefindAPI().then(function (pagefind) {
    if (!pagefind) {
      return [];
    }

    return pagefind.search(query).then(function (searchResults) {
      if (!searchResults || !searchResults.results) {
        return [];
      }

      // Get top 10 keyword results with their data
      var promises = searchResults.results.slice(0, 10).map(function (result, index) {
        return result.data().then(function (data) {
          return {
            url: data.url,
            title: data.meta && data.meta.title ? data.meta.title : "Untitled",
            excerpt: data.excerpt || "",
            rank: index + 1,
            source: "keyword"
          };
        });
      });

      return Promise.all(promises);
    }).catch(function () {
      return [];
    });
  });

  var semanticPromise = new Promise(function (resolve) {
    if (!window.VectorSearch.isAvailable()) {
      resolve([]);
      return;
    }

    window.VectorSearch.search(query, 10, 0.25).then(function (results) {
      resolve(results.map(function (result, index) {
        return {
          url: result.chunk.url,
          title: result.chunk.title,
          section: result.chunk.section,
          excerpt: result.chunk.text,
          categories: result.chunk.categories,
          similarity: result.similarity,
          rank: index + 1,
          source: "semantic"
        };
      }));
    }).catch(function () {
      resolve([]);
    });
  });

  // Combine results using Reciprocal Rank Fusion (RRF)
  Promise.all([keywordPromise, semanticPromise]).then(function (allResults) {
    var keywordResults = allResults[0];
    var semanticResults = allResults[1];

    var combined = combineResultsRRF(keywordResults, semanticResults);
    renderHybridResults(combined);
  });
}

function combineResultsRRF(keywordResults, semanticResults) {
  var k = 60; // RRF constant
  var scores = {};
  var dataByUrl = {};

  // Score keyword results
  keywordResults.forEach(function (result) {
    var url = result.url;
    if (!scores[url]) scores[url] = 0;
    scores[url] += 1 / (k + result.rank);

    if (!dataByUrl[url]) {
      dataByUrl[url] = {
        url: url,
        title: result.title,
        excerpt: result.excerpt,
        sources: []
      };
    }
    dataByUrl[url].sources.push("keyword");
    dataByUrl[url].keywordRank = result.rank;
  });

  // Score semantic results
  semanticResults.forEach(function (result) {
    var url = result.url;
    if (!scores[url]) scores[url] = 0;
    scores[url] += 1 / (k + result.rank);

    if (!dataByUrl[url]) {
      dataByUrl[url] = {
        url: url,
        title: result.title,
        excerpt: result.excerpt,
        sources: []
      };
    }
    dataByUrl[url].sources.push("semantic");
    dataByUrl[url].semanticRank = result.rank;
    dataByUrl[url].similarity = result.similarity;
    dataByUrl[url].section = result.section;
    dataByUrl[url].categories = result.categories;

    // Prefer semantic excerpt if available
    if (result.excerpt && result.excerpt.length > 0) {
      dataByUrl[url].excerpt = result.excerpt;
    }
  });

  // Convert to sorted array
  var combined = Object.keys(scores).map(function (url) {
    var data = dataByUrl[url];
    data.rrfScore = scores[url];
    return data;
  });

  combined.sort(function (a, b) {
    return b.rrfScore - a.rrfScore;
  });

  return combined.slice(0, 10);
}

function renderHybridResults(results) {
  var $container = $(".hybrid-search__results");
  var $empty = $(".hybrid-search__empty");

  $container.empty();

  if (results.length === 0) {
    $empty.show();
    return;
  }

  $empty.hide();

  results.forEach(function (result, index) {
    // Calculate display score (normalize RRF to percentage-like value)
    var displayScore = Math.min(99, Math.round(result.rrfScore * 3000));

    var $result = $('<a class="vector-search__result" href="' + result.url + '"></a>');
    $result.css("animation-delay", (index * 0.04) + "s");

    // Header row: title + score + source indicators
    var $header = $('<div class="vector-search__result-header"></div>');

    var $title = $('<span class="vector-search__result-title"></span>').text(result.title);

    var $score = $('<div class="vector-search__result-score"></div>');
    var scoreClass = displayScore >= 70 ? "high" : displayScore >= 50 ? "medium" : "low";
    $score.addClass("vector-search__result-score--" + scoreClass);
    $score.html('<span class="score-value">' + displayScore + '</span><span class="score-percent">%</span>');

    $header.append($title, $score);

    // Meta row: section + source badges
    var $meta = $('<div class="vector-search__result-meta"></div>');

    if (result.section && result.section !== result.title) {
      var $section = $('<span class="vector-search__result-section"></span>');
      $section.html('<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h7"/></svg>' + result.section);
      $meta.append($section);
    }

    // Source badges showing which search found this
    var $sources = $('<div class="hybrid-search__sources"></div>');
    if (result.sources.indexOf("keyword") !== -1) {
      $sources.append('<span class="hybrid-search__source hybrid-search__source--keyword">K</span>');
    }
    if (result.sources.indexOf("semantic") !== -1) {
      $sources.append('<span class="hybrid-search__source hybrid-search__source--semantic">S</span>');
    }
    $meta.append($sources);

    // Excerpt with truncation
    var excerptText = result.excerpt || "";
    // Strip HTML tags from Pagefind excerpts
    excerptText = excerptText.replace(/<[^>]*>/g, "");
    excerptText = excerptText.length > 150 ? excerptText.substring(0, 150) + "..." : excerptText;
    var $excerpt = $('<p class="vector-search__result-excerpt"></p>').text(excerptText);

    // Categories footer
    var $footer = $('<div class="vector-search__result-footer"></div>');
    if (result.categories && result.categories.length > 0) {
      var $categories = $('<div class="vector-search__result-categories"></div>');
      result.categories.slice(0, 3).forEach(function (cat) {
        $categories.append(
          $('<span class="vector-search__result-category"></span>').text(cat)
        );
      });
      $footer.append($categories);
    }

    $result.append($header, $meta, $excerpt, $footer);
    $container.append($result);
  });
}

function openSearch() {
  var $modal = $(".search-modal");
  if ($modal.length === 0 || $modal.hasClass("is-active")) return;

  // Prevent body scroll
  $("body").addClass("search-modal--open");

  // Show modal
  $modal.addClass("is-active");

  // Initialize Pagefind UI on first open (lazy load)
  if (!window.pagefindInitialized && typeof PagefindUI !== "undefined") {
    new PagefindUI({
      element: "#pagefind-container",
      showSubResults: true,
      showImages: false,
      excerptLength: 20,
    });
    window.pagefindInitialized = true;
  }

  // Ensure correct search mode is active (handles mobile default to keyword)
  // Force init on first open to ensure correct container is shown
  switchSearchMode(searchMode, true);

  // Trap focus within modal
  trapFocus($modal[0]);
}

function closeSearch() {
  var $modal = $(".search-modal");
  if (!$modal.hasClass("is-active")) return;

  // Add closing animation class
  $modal.addClass("is-closing");

  // Restore body scroll
  $("body").removeClass("search-modal--open");

  // Remove classes after animation
  setTimeout(function () {
    $modal.removeClass("is-active is-closing");

    // Clean up focus trap
    if ($modal[0]._trapFocusCleanup) {
      $modal[0]._trapFocusCleanup();
    }
  }, 250);
}

/*-------------------------------------------------------------------------*/
/* KEYBOARD NAVIGATION (Vim/GitHub style: g + key)                         */
/*-------------------------------------------------------------------------*/

function initKeyboardNav() {
  var gKeyPressed = false;
  var gKeyTimeout = null;

  $(document).on("keydown", function (e) {
    // Ignore if user is typing in an input/textarea
    var tag = e.target.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea" || e.target.isContentEditable) {
      return;
    }

    // Ignore if any modifier keys are held (except shift)
    if (e.metaKey || e.ctrlKey || e.altKey) {
      return;
    }

    var key = e.key.toLowerCase();

    // First key: 'g' starts the sequence
    if (key === "g" && !gKeyPressed) {
      gKeyPressed = true;

      // Reset after 1.5 seconds if no second key
      clearTimeout(gKeyTimeout);
      gKeyTimeout = setTimeout(function () {
        gKeyPressed = false;
      }, 1500);

      return;
    }

    // Second key: execute action if 'g' was pressed
    if (gKeyPressed) {
      gKeyPressed = false;
      clearTimeout(gKeyTimeout);

      switch (key) {
        case "h": // g h → Home
          e.preventDefault();
          window.location.href = "/";
          break;

        case "c": // g c → Categories
          e.preventDefault();
          window.location.href = "/categories/";
          break;

        case "a": // g a → About
          e.preventDefault();
          window.location.href = "/about/";
          break;

        case "o": // g o → Open contact modal
          e.preventDefault();
          openContactHub();
          break;

        case "s": // g s → Open search
          e.preventDefault();
          openSearch();
          break;
      }
    }

    // Standalone shortcuts (no 'g' prefix needed)
    if (!gKeyPressed) {
      // '/' opens search (common pattern)
      if (key === "/" && !$(".search-modal").hasClass("is-active")) {
        e.preventDefault();
        openSearch();
      }

      // '?' opens shortcuts help
      if ((key === "?" || (e.shiftKey && key === "/")) && !$(".shortcuts-modal").hasClass("is-active")) {
        e.preventDefault();
        openShortcuts();
      }
    }
  });
}

/*-------------------------------------------------------------------------*/
/* SHORTCUTS MODAL                                                         */
/*-------------------------------------------------------------------------*/

function initShortcuts() {
  var $modal = $(".shortcuts-modal");
  var $trigger = $(".js-shortcuts-trigger");
  var $close = $(".js-shortcuts-close");

  if ($modal.length === 0) return;

  // Click trigger to open
  $trigger.on("click", function (e) {
    e.preventDefault();
    openShortcuts();
  });

  // Click close button or overlay to close
  $close.on("click", function (e) {
    e.preventDefault();
    closeShortcuts();
  });
}

function openShortcuts() {
  var $modal = $(".shortcuts-modal");
  if ($modal.length === 0 || $modal.hasClass("is-active")) return;

  // Close other modals first
  closeSearch();
  closeContactHub();

  $("body").addClass("shortcuts-modal--open");
  $modal.addClass("is-active");
}

function closeShortcuts() {
  var $modal = $(".shortcuts-modal");
  if (!$modal.hasClass("is-active")) return;

  $modal.addClass("is-closing");
  $("body").removeClass("shortcuts-modal--open");

  setTimeout(function () {
    $modal.removeClass("is-active is-closing");
  }, 200);
}

// Add to initialization
$(document).ready(function () {
  // ... existing code runs first via the other ready block
  initKeyboardNav();
  initShortcuts();
  initGitHubReposDrawer();
});

/*-------------------------------------------------------------------------*/
/* GITHUB REPOS DRAWER                                                      */
/* -----------------------------------------------------------------------*/

// Repos to display
var FAVORITE_REPOS = [
  "johnlarkin1/claude-code-extensions",
  "johnlarkin1/imessage-data-foundry",
  "johnlarkin1/larkin-mcp",
  "johnlarkin1/yourname-mcp",
  "johnlarkin1/word-hunt-solver",
];

// Cache for GitHub API responses (5-minute TTL)
var reposCache = {
  data: null,
  timestamp: null,
  ttl: 5 * 60 * 1000, // 5 minutes
};

// Language colors (reused from initGitHubRepoCards)
var drawerLanguageColors = {
  JavaScript: "#f1e05a",
  TypeScript: "#2b7489",
  Python: "#3572A5",
  Java: "#b07219",
  "C++": "#f34b7d",
  C: "#555555",
  Go: "#00ADD8",
  Rust: "#dea584",
  Ruby: "#701516",
  Shell: "#89e051",
  HTML: "#e34c26",
  CSS: "#563d7c",
};

function initGitHubReposDrawer() {
  var $drawer = $(".gh-repos-drawer");
  var $trigger = $(".js-gh-repos-trigger");
  var $close = $(".js-gh-repos-drawer-close");
  var $retry = $(".js-gh-repos-retry");

  if ($drawer.length === 0) return;

  // Open drawer on trigger click
  $trigger.on("click", function (e) {
    e.preventDefault();
    e.stopPropagation(); // Prevent dropdown from closing
    openGitHubReposDrawer();
  });

  // Close drawer
  $close.on("click", function (e) {
    e.preventDefault();
    closeGitHubReposDrawer();
  });

  // Retry button
  $retry.on("click", function (e) {
    e.preventDefault();
    fetchGitHubRepos(true); // Force refresh
  });
}

function openGitHubReposDrawer() {
  var $drawer = $(".gh-repos-drawer");
  if ($drawer.length === 0 || $drawer.hasClass("is-active")) return;

  // Close other modals first
  closeSearch();
  closeContactHub();
  closeShortcuts();

  // Prevent body scroll
  $("body").addClass("gh-repos-drawer--open");

  // Show drawer
  $drawer.addClass("is-active");

  // Fetch repos if needed
  fetchGitHubRepos();

  // Set up focus trapping
  trapFocus($drawer[0]);
}

function closeGitHubReposDrawer() {
  var $drawer = $(".gh-repos-drawer");
  if (!$drawer.hasClass("is-active")) return;

  // Add closing class for exit animation
  $drawer.addClass("is-closing");

  // Wait for animation then hide
  setTimeout(function () {
    $drawer.removeClass("is-active is-closing");
    $("body").removeClass("gh-repos-drawer--open");

    // Clean up focus trap
    if ($drawer[0]._trapFocusCleanup) {
      $drawer[0]._trapFocusCleanup();
    }

    // Return focus to trigger
    $(".js-gh-repos-trigger").first().focus();
  }, 250);
}

function fetchGitHubRepos(forceRefresh) {
  var $loading = $(".gh-repos-drawer__loading");
  var $error = $(".gh-repos-drawer__error");
  var $list = $(".gh-repos-drawer__list");

  // Check cache first
  if (!forceRefresh && reposCache.data && reposCache.timestamp) {
    var age = Date.now() - reposCache.timestamp;
    if (age < reposCache.ttl) {
      $loading.hide();
      renderGitHubRepos(reposCache.data);
      return;
    }
  }

  // Show loading, hide others
  $loading.show();
  $error.hide();
  $list.empty();

  // Fetch all repos in parallel
  var promises = FAVORITE_REPOS.map(function (repo) {
    return fetch("https://api.github.com/repos/" + repo)
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .catch(function (err) {
        console.warn("Failed to fetch " + repo + ":", err);
        return null; // Return null for failed requests
      });
  });

  Promise.all(promises)
    .then(function (results) {
      // Filter out failed requests
      var repos = results.filter(function (repo) {
        return repo !== null && !repo.message;
      });

      if (repos.length === 0) {
        throw new Error("No repos could be fetched");
      }

      // Cache the results
      reposCache.data = repos;
      reposCache.timestamp = Date.now();

      $loading.hide();
      renderGitHubRepos(repos);
    })
    .catch(function (err) {
      console.error("GitHub API error:", err);
      $loading.hide();
      $error.show();
    });
}

function renderGitHubRepos(repos) {
  var $list = $(".gh-repos-drawer__list");
  $list.empty();

  repos.forEach(function (repo) {
    var starsFormatted = (repo.stargazers_count || 0).toLocaleString();
    var languageColor = drawerLanguageColors[repo.language] || "#ccc";

    var $card = $(
      '<a class="gh-repos-drawer__repo-card" href="' +
        repo.html_url +
        '" target="_blank" rel="noopener noreferrer"></a>'
    );

    // Header with name and stars
    var $header = $('<div class="gh-repos-drawer__repo-header"></div>');
    var $name = $(
      '<h3 class="gh-repos-drawer__repo-name">' + repo.name + "</h3>"
    );
    var $stars = $(
      '<span class="gh-repos-drawer__repo-stars">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>' +
        starsFormatted +
        "</span>"
    );
    $header.append($name, $stars);

    // Description
    var $description = $(
      '<p class="gh-repos-drawer__repo-description">' +
        (repo.description || "No description available") +
        "</p>"
    );

    // Meta (language)
    var $meta = $('<div class="gh-repos-drawer__repo-meta"></div>');
    if (repo.language) {
      var $language = $(
        '<span class="gh-repos-drawer__repo-language">' +
          '<span class="gh-repos-drawer__language-dot" style="background-color: ' +
          languageColor +
          '"></span>' +
          repo.language +
          "</span>"
      );
      $meta.append($language);
    }

    // Topics
    var $topics = $('<div class="gh-repos-drawer__repo-topics"></div>');
    if (repo.topics && repo.topics.length > 0) {
      repo.topics.slice(0, 4).forEach(function (topic) {
        $topics.append(
          '<span class="gh-repos-drawer__repo-topic">' + topic + "</span>"
        );
      });
    }

    $card.append($header, $description, $meta);
    if (repo.topics && repo.topics.length > 0) {
      $card.append($topics);
    }

    $list.append($card);
  });
}
