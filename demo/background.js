// create context menu
var contextMenuItem = {
	"id": "sentimentAnalysis",
	"title": "Perform Sentiment Analysis",
	"contexts": ["selection"]
}
chrome.contextMenus.create(contextMenuItem)


chrome.contextMenus.onClicked.addListener(function(clickData) {
	if (clickData.menuItemId == "sentimentAnalysis" && clickData.selectionText) {
		var text = clickData.selectionText		
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
	}
})