let count =['445 - x%', '31 - x%', '27 - x%', '21 - x%', '19 - x%', '15 - x%', '15 - x%', '10 - x%']
let x_cust =[1.3546330311235595, 1.816908228067302, -1.810794809119315, -4.60517018598809, -4.60517018598809, 1.336012636110227, 1.880424979192522, 1.8748338864907985]
let y_cust = ['customer service', 'customer service reps', 'comcast customer', 'worst customer service', 'new customer', 'customer service representatives', 'great customer service', 'customer service team']

var greaterThan = document.getElementById('cust'),
      data = [{
        type: 'bar',
				"x": x_cust, 
				"y": y_cust,
		text: count.map(String),
		textposition: 'auto',
 		hoverinfo: 'none',
        orientation: 'h',
        mode: 'markers',
        marker: {
          color: ['rgba(0,0,255,0.61)','rgba(0,0,255,0.85)','rgba(255,0,0,0.43)','rgba(255,0,0,0.1)','rgba(255,0,0,0.1)','rgba(0,0,255,0.56)','rgba(0,0,255,1)','rgba(0,0,255,0.90)']
        },
      }];

layout = {hovermode:'closest',
          title:'Containing Customer',
          xaxis:{zeroline:false, hoverformat: '.2f', title: 'Log Odds'}
     }

Plotly.newPlot('cust', data, layout, {responsive: true});