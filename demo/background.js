// create context menu
var contextMenuItem = {
	"id": "sentimentAnalysis",
	"title": "Perform Sentiment Analysis",
	"contexts": ["selection"]
}
chrome.contextMenus.create(contextMenuItem)

chrome.contextMenus.onClicked.addListener(function(clickData) {
	if (clickData.menuItemId == "sentimentAnalysis" && clickData.selectionText) {
		chrome.storage.sync.set({'text': clickData.selectionText}, null);
	}
})