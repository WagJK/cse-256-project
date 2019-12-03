$(function() {
	chrome.storage.sync.get(['text', 'score'], function(obj) {
		$('#text').text(obj.text);
		$('#score').text(obj.score);
		var label = "?";
		if (obj.score < 0.1) {
			label = "Very Negative"
			button_color = "badge-danger"
			sub_label = "You mad?"
		} else if (obj.score < 0.35) {
			label = "Negative"
			button_color = "badge-danger"
			sub_label = "Maybe change your tone to be a bit nicer."
		} else if (obj.score < 0.65) {
			label = "Neutral"
			button_color = "badge-warning"
			sub_label = "Not positive, not negative, just neutral."
		} else if (obj.score < 0.9) {
			label = "Positive"
			button_color = "badge-success"
			sub_label = "You seem happy."
		} else {
			label = "Very Positive"
			button_color = "badge-success"
			sub_label = "You are having a blast!"
		} 
		$('#label').text(label)
		$('#label').addClass(button_color)
		$('#sublabel').text(sub_label)
		console.log(obj);
	})
});

$('#analyzeButton').click(function analyze() {
	var text = $('#text').val();
	// ajax request
	console.log(text);
	var form = new FormData();
	form.append('text', text);
	var settings = {
		"async": true,
		"crossDomain": true,
		"url": "http://127.0.0.1:5000/",
		"method": "POST",
		"processData": false,
		"contentType": false,
		"mimeType": "multipart/form-data",
		"data": form
	}
	$.ajax(settings).done(function (response) {
		var parsed_response = JSON.parse(response)
		console.log(parsed_response);
		chrome.storage.sync.set({'text': text}, null);
		chrome.storage.sync.set({'score': parsed_response.sentiment}, null);
	});
});
