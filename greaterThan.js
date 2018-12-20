let x = [445, 120, 41, 36, 31, 27, 26, 23, 19, 19, 19] 
//let odds = [1.3546330311235595, 1.384733829234302, 1.2412821181688312, 0.9046614144180127, 1.8169082280673023, 1.8626980518843244, 2.703514734926804, 2.5342314554919816, 2.257018930189952, 2.257018930189952, 1.7968277890993434]
let y = ["customer service", "internet speed", "comcast internet service", "long time", "customer service reps", "made sure", "installation process", "bad weather", "timely manner", "pretty good", "high speed"]

var greaterThan = document.getElementById('oddsGreater'),
      data = [{
        type: 'bar',
				"x": x, 
				"y": y,
        orientation: 'h',
        mode: 'markers',
        marker: {
          color: ['rgba(20,20,250,0.25)','rgba(20,20,250,0.26)','rgba(20,20,250,0.18)','rgba(20,20,250,0.1)','rgba(20,20,250,0.50)','rgba(20,20,250,0.53)','rgba(20,20,250,1)','rgba(20,20,250,0.90)','rgba(20,20,250,0.75)','rgba(20,20,250,0.75)','rgba(20,20,250,0.49)']
        },
      }];

layout = {hovermode:'closest',
          title:'Positive Odds',
          xaxis:{zeroline:false, hoverformat: '.2f', title: 'Count'}
     }

Plotly.newPlot('oddsGreater', data, layout, {responsive: true});