window.jQuery = window.$ = require("jquery");
require("velocity-animate/velocity.js");
require("lazysizes");
require("lazysizes/plugins/unveilhooks/ls.unveilhooks.js");

// Jquery & Velocity JS included in GULP
$(document).ready(function () {
  toggleMobileNav();
  ShowHideNav();
  formCheck();
  filterBlogCategories();
  initGitHubRepoCards();
  initLightbox();
  initMermaid();
  initScrollIndicator();
});

// Close modal if ESC is pressed
$(document).keyup(function (e) {
  if (e.keyCode === 27) {
    removeModal();
    closeLightbox();
  }
});

$(window).resize(function () {
  $(".header").removeClass("hide-nav"); // Ensure nav will be shown on resize
  $(".header__toggle").removeClass("--open");
  $(".header__links").removeClass("js--open");
  $(".header__links").removeAttr("style"); // If mobile nav was collapsed, make sure it's show on DESK
  $(".header__overlay").remove(); // Remove mobile navigation overlay in case it was opened
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
    },
    progress: function () {
      $(".header__overlay").addClass("--open");
    },
    complete: function () {
      $(this).addClass("js--open");
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
    },
    progress: function () {
      $(".header__overlay").removeClass("--open");
    },
    complete: function () {
      $(this).removeClass("js--open");
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

/*-------------------------------------------------------------------------*/
/* MERMAID DIAGRAM INITIALIZATION */
/* -----------------------------------------------------------------------*/

function initMermaid() {
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
    const heroHeight = hero.length > 0 ? hero.offset().top + hero.outerHeight() : 0;

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
