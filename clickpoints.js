var myPlot = document.getElementById('clickPoints'),
    data = [{
      "x": [445, 120, 133, 26, 23, 11, 66, 31, 19, 19, 27, 17, 46, 10, 11, 19, 41, 14, 15, 15, 10, 10, 10, 64, 31, 42, 29, 28, 27, 18, 26, 26, 25, 24, 24, 10, 10, 10, 10, 34, 23, 33, 22, 22, 22, 32, 21, 21, 21, 21], "y": [1.3546330311235595, 1.384733829234302, -2.787636633811688, 2.703514734926804, 2.5342314554919816, 3.7822761290544826, -4.605170185988091, 1.8169082280673023, 2.257018930189952, 2.257018930189952, 1.8626980518843244, 2.3575779840039357, -4.605170185988091, 2.8610670738065425, 2.455097939718171, 1.7968277890993434, 1.2412821181688312, 2.0632514289849064, 1.8804249791925227, 1.8804249791925227, 2.3193663595677965, 2.3193663595677965, 2.3193663595677965, -1.578630519917741, -4.605170185988091, -2.273952832385637, -4.605170185988091, -4.605170185988091, -4.605170185988091, 1.4734436206011003, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091, 1.8748338864907985, 1.8748338864907985, 1.8748338864907985, 1.8748338864907985, -2.0527836530120442, -4.605170185988091, -2.021500092726019, -4.605170185988091, -4.605170185988091, -4.605170185988091, -1.9892397547075746, -4.605170185988091, -4.605170185988091, -4.605170185988091, -4.605170185988091], "text": ["customer service", "internet speed", "called comcast", "installation process", "bad weather", "would definitely recommend", "called back", "customer service reps", "timely manner", "pretty good", "made sure", "great service", "worst company", "explained everything", "really good", "high speed", "comcast internet service", "need help", "good service", "great customer service", "would highly recommend", "super fast", "always fast", "next day", "next month", "new modem", "contacted comcast", "finally got", "comcast service", "one time", "cable service", "would need", "get away", "ever dealt", "3 months", "job done", "pretty fast", "much faster", "customer service team", "call comcast", "stay away", "phone number", "would receive", "phone calls", "bad service", "never received", "last week", "worst customer service", "come back", "week later"],
      mode: 'markers+text',
      name: 'Markers and Text',
      textposition: 'bottom',
      type: 'scatter',
      hoverinfo:"x+y"

    }];

layout = {hovermode:'closest',
          title:'Click on Points',
          xaxis:{zeroline:false, hoverformat: '.0f', title: '# Reviewers',type: 'log',autorange: true},
          yaxis:{zeroline:false, hoverformat: '.2f', title: 'Log-Odds of Positive Review'}
     }

Plotly.newPlot('clickPoints', data, layout);

//Martin
myPlot.on('plotly_click', function(data){
    console.log(data.points[0])
});
