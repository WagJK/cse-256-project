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
		chrome.storage.sync.set({'text': text}, null);
		
		// ajax request
		var xhttp = new XMLHttpRequest();
		xhttp.onreadystatechange = function() {
			if (this.readyState == 4 && this.status == 200) {
				document.getElementById("demo").innerHTML = this.responseText;
			}
		};
		xhttp.open("GET", "ajax_info.txt", true);
		xhttp.send();
	}
})