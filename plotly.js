var myPlot = document.getElementById('myDiv'),
    data = [{
      "x": [25, 74, 20, 50, 25, 43, 35, 10, 47, 21, 17, 18, 16, 30, 37, 14, 26, 31, 25, 24, 12, 33, 10, 17, 46, 20, 38, 26, 16, 23], "y": [-2.280433199571406, 1.8792207429396666, -2.6210809800648263, 2.4097291940104224, -1.8286638229471803, 2.241372529891609, -1.415696159860169, -3.058994244708425, 1.5922411202368953, -1.563411901284539, -1.7417312296661032, 4.605170185988092, -1.651289322181948, -1.1402299307071682, 1.312374021334796, 4.605170185988092, 1.6802706084809809, 1.4315899257022147, 1.6362262631267674, 1.5902919709278749, 4.605170185988092, 1.1771563671322829, -1.6994101749676764, -1.2131937670181316, -0.6939977631900378, -1.0577055773289963, -0.7598214505530844, 1.2274337490052563, 1.8934462864626582, -0.9445655934718953], "text": ["service call", "claim online", "service request", "companyname online", "called back", "took care", "customer service", "customer service rep", "good job", "contacted companyname", "finally got", "working great", "second opinion", "home warranty", "good experience", "recommended companyname", "great job", "working fine", "really good", "taken care", "really nice", "went online", "contractor said", "service fee", "water heater", "20 minutes", "call companyname", "air conditioning", "went ahead", "30 minutes"],
      mode: 'markers+text',
      name: 'Markers and Text',
      textposition: 'bottom',
      type: 'scatter',
      hoverinfo:"x+y"

    }];

layout = {hovermode:'closest',
          title:'Click on Points',
          xaxis:{zeroline:false, hoverformat: '.0f', title: '# Reviewers'},
          yaxis:{zeroline:false, hoverformat: '.2f', title: 'Log-Odds of Positive Review'}
     }

Plotly.newPlot('myDiv', data, layout, {responsive: true});

myPlot.on('plotly_click', function(data){
    console.log(data.points[0])
});