let x_small = [133, 66, 64, 46, 42, 39, 35, 34, 33, 32, 31, 30, 29, 28, 27, 27, 27, 26, 26, 25, 24, 24, 23, 22, 22, 22, 21, 21, 21, 21, 21, 20, 20, 19, 19, 19, 19, 18, 18]
//let odds = [-2.787636633811688, -4.605170185988091, -1.578630519917741, -4.605170185988091, -2.273952832385637, -1.4739255038634194, -1.3574649220200974, -2.0527836530120442, -2.021500092726019, -1.9892397547075746, -4.605170185988091, -1.9215258854788604, -4.605170185988091, -4.605170185988091, -1.8107948091193158, -1.8107948091193158, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091]
let y_small = ["called comcast", "called back", "next day", "worst company", "new modem", "6 months", "first time", "call comcast", "phone number", "never received", "next month", "tech support", "contacted comcast", "finally got", "service call", "comcast customer", "comcast service", "would need", "cable service", "get away", "ever dealt", "3 months", "stay away", "would receive", "phone calls", "bad service", "second time", "week later", "worst customer service", "come back", "last week", "cable box", "came back", "new customer", "poor service", "one showed", "month later", "account number", "never showed"]

var smallerThan = document.getElementById('oddsSmaller'),
      data = [{
        type: 'bar',
        "x": x_small,
        "y": y_small,
        orientation: 'h',
        mode: 'markers',
        marker: {
          color: ['rgba(250,20,20,0.55)','rgba(250,20,20,0.1)','rgba(250,20,20,0.93)','rgba(250,20,20,0.1)','rgba(250,20,20,0.71)','rgba(250,20,20,0.96)','rgba(250,20,20,1)','rgba(250,20,20,0.78)','rgba(250,20,20,0.79)','rgba(250,20,20,0.80)','rgba(250,20,20,0.1)','rgba(250,20,20,0.82)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.86)','rgba(250,20,20,0.86)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)','rgba(250,20,20,0.1)']
        },
      }];

layout = {hovermode:'closest',
          title:'Negative Odds',
          xaxis:{zeroline:false, hoverformat: '.2f', title: 'Count'}
     }

Plotly.newPlot('oddsSmaller', data, layout, {responsive: true});