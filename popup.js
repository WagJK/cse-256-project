$(function() {
	chrome.storage.sync.get(['text'], function(obj) {
		$('#text').text(obj.text);
	})
});