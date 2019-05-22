const width_threshold = 750;

function drawLineChart() {
  if ($("#lineChart").length) {
    ctxLine = document.getElementById("lineChart").getContext("2d");
    optionsLine = {
      scales: {
        yAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: "Accuracy (%)"
            }
          }
        ]
      }
    };

    // Set aspect ratio based on window width
    optionsLine.maintainAspectRatio =
      $(window).width() < width_threshold ? false : true;

    configLine = {
      type: "line",
      data: {
        labels: [
          "1/6 of Records",
          "1/5 of Records",
          "1/4 of Records",
          "1/3 of Records"
        ],
        datasets: [
          {
            label: "KNeighborsClassifier",
            data: [38.79, 37.51, 39.77, 38.57],
            fill: false,
            borderColor: "rgb(75, 192, 192)",
            lineTension: 0.1
          },
          {
            label: "SVC",
            data: [38.13, 39.69, 41.29, 39.78],
            fill: false,
            borderColor: "rgba(255,99,132,1)",
            lineTension: 0.1
          },
          {
            label: "LinearSVC",
            data: [44.74, 47.07, 47.12, 46.14],
            fill: false,
            borderColor: "rgba(153, 102, 255, 1)",
            lineTension: 0.1
          },
          {
            label: "NuSVC",
            data: [39.48, 41.97, 42.15, 40.65],
            fill: false,
            borderColor:"rgba(133, 43, 200)",
            lineTension: 0.1
          },
          {
            label: "DecisionTreeClassifier",
            data: [37.41, 39.64, 40.75, 38.99],
            fill: false,
            borderColor:"rgba(200, 200, 55)",
            lineTension: 0.1
          },
          {
            label: "RandomForestClassifier",
            data: [40.92, 42.92, 42.73, 41.85],
            fill: false,
            borderColor: "rgba(200, 100, 30)",
            lineTension: 0.1
          },
          {
            label: "AdaBoostClassifier",
            data: [46.48, 47, 46.16, 46.61],
            fill: false,
            borderColor:"rgba(255, 10, 55)",
            lineTension: 0.1
          },
          {
            label: "GradientBoostingClassifier",
            data: [46.72, 47.67, 47.44, 46.71],
            fill: false,
            borderColor: "rgba(60, 255, 30)",
            lineTension: 0.1
          },
          {
            label: "GaussianNB",
            data: [40.11, 40.89,40.01, 41.13],
            fill: false,
            borderColor:"rgba(150, 15, 60)",
            lineTension: 0.1
          },
          {
            label: "LinearDiscriminantAnalysis",
            data: [45.61,45.2, 45.32, 46.64],
            fill: false,
            borderColor: "rgba(175, 30, 45)",
            lineTension: 0.1
          },
          {
            label: "QuadraticDiscriminantAnalysis",
            data: [39.21, 27.70, 37.59, 36.24],
            fill: false,
            borderColor:"rgba(0, 156, 95)",
            lineTension: 0.1
          },
          {
            label: "FastText",
            data: [36.83, 38.71, 41.53, 43.22],
            fill: false,
            borderColor: "rgba(200, 35, 95)",
            lineTension: 0.1
          }
          
        ]
      },
      options: optionsLine
    };

    lineChart = new Chart(ctxLine, configLine);
  }
}

// function drawBarChart() {
//   if ($("#barChart").length) {
//     ctxBar = document.getElementById("barChart").getContext("2d");

//     optionsBar = {
//       responsive: true,
//       scales: {
//         yAxes: [
//           {
//             ticks: {
//               beginAtZero: true
//             },
//             scaleLabel: {
//               display: true,
//               labelString: "Accuracy (%)"
//             }
//           }
//         ]
//       }
//     };

//     optionsBar.maintainAspectRatio =
//       $(window).width() < width_threshold ? false : true;

//     configBar = {
//       type: "bar",
//       data: {
//         labels: ["Red", "Blue", "Yellow", "Green", "Purple", "Orange"],
//         datasets: [
//           {
//             label: "# of Hits",
//             data: [12, 19, 3, 5, 2, 3],
//             backgroundColor: [
//               "rgba(255, 99, 132, 0.2)",
//               "rgba(54, 162, 235, 0.2)",
//               "rgba(255, 206, 86, 0.2)",
//               "rgba(75, 192, 192, 0.2)",
//               "rgba(153, 102, 255, 0.2)",
//               "rgba(255, 159, 64, 0.2)"
//             ],
//             borderColor: [
//               "rgba(255,99,132,1)",
//               "rgba(54, 162, 235, 1)",
//               "rgba(255, 206, 86, 1)",
//               "rgba(75, 192, 192, 1)",
//               "rgba(153, 102, 255, 1)",
//               "rgba(255, 159, 64, 1)"
//             ],
//             borderWidth: 1
//           }
//         ]
//       },
//       options: optionsBar
//     };

//     barChart = new Chart(ctxBar, configBar);
//   }
// }

function drawPieChart() {
  if ($("#pieChart").length) {
    ctxPie = document.getElementById("pieChart").getContext("2d");
    optionsPie = {
      responsive: true,
      maintainAspectRatio: false
    };

    configPie = {
      type: "pie",
      data: {
        datasets: [
          {
            data: [4600, 5400],
            backgroundColor: [
              window.chartColors.purple,
              window.chartColors.green
            ],
            label: "Storage"
          }
        ],
        labels: ["Used: 4,600 GB", "Available: 5,400 GB"]
      },
      options: optionsPie
    };

    pieChart = new Chart(ctxPie, configPie);
  }
}

function updateChartOptions() {
  if ($(window).width() < width_threshold) {
    if (optionsLine) {
      optionsLine.maintainAspectRatio = false;
    }
    if (optionsBar) {
      optionsBar.maintainAspectRatio = false;
    }
  } else {
    if (optionsLine) {
      optionsLine.maintainAspectRatio = true;
    }
    if (optionsBar) {
      optionsBar.maintainAspectRatio = true;
    }
  }
}

function updateLineChart() {
  if (lineChart) {
    lineChart.options = optionsLine;
    lineChart.update();
  }
}

function updateBarChart() {
  if (barChart) {
    barChart.options = optionsBar;
    barChart.update();
  }
}

function reloadPage() {
  setTimeout(function() {
    window.location.reload();
  }); // Reload the page so that charts will display correctly
}

function drawCalendar() {
  if ($("#calendar").length) {
    $("#calendar").fullCalendar({
      height: 400,
      events: [
        {
          title: "Meeting",
          start: "2018-09-1",
          end: "2018-09-2"
        },
        {
          title: "Marketing trip",
          start: "2018-09-6",
          end: "2018-09-8"
        },
        {
          title: "Follow up",
          start: "2018-10-12"
        },
        {
          title: "Team",
          start: "2018-10-17"
        },
        {
          title: "Company Trip",
          start: "2018-10-25",
		  end: "2018-10-27"
        },
        {
          title: "Review",
          start: "2018-11-12"
        },
        {
          title: "Plan",
          start: "2018-11-18"
        }
      ],
      eventColor: "rgba(54, 162, 235, 0.4)"
    });
  }
}
