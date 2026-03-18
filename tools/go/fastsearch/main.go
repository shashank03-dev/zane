package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
	Source  string `json:"source"`
}

var (
	linkRe = regexp.MustCompile(`<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)
	tagRe  = regexp.MustCompile(`<[^>]+>`)
)

func cleanTitle(s string) string {
	noTags := tagRe.ReplaceAllString(s, "")
	decoded := html.UnescapeString(noTags)
	return strings.TrimSpace(decoded)
}

func normalizeURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}

	if strings.Contains(u.Host, "duckduckgo.com") && strings.HasPrefix(u.Path, "/l/") {
		uddg := u.Query().Get("uddg")
		if uddg != "" {
			decoded, err := url.QueryUnescape(uddg)
			if err == nil {
				return decoded
			}
			return uddg
		}
	}

	return raw
}

func searchDuckDuckGo(query string, maxResults int, timeoutSec int) ([]SearchResult, error) {
	client := &http.Client{Timeout: time.Duration(timeoutSec) * time.Second}

	reqURL := "https://duckduckgo.com/html/?q=" + url.QueryEscape(query)
	req, err := http.NewRequest(http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "zane-fastsearch/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("search request failed with status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	matches := linkRe.FindAllStringSubmatch(string(body), maxResults)
	results := make([]SearchResult, 0, len(matches))
	for _, match := range matches {
		if len(match) < 3 {
			continue
		}
		results = append(results, SearchResult{
			Title:   cleanTitle(match[2]),
			URL:     normalizeURL(match[1]),
			Snippet: "",
			Source:  "go-fastsearch",
		})
	}

	return results, nil
}

func main() {
	query := flag.String("query", "", "Search query")
	maxResults := flag.Int("max-results", 5, "Maximum number of results")
	timeoutSec := flag.Int("timeout", 15, "HTTP timeout in seconds")
	flag.Parse()

	if strings.TrimSpace(*query) == "" {
		fmt.Fprintln(os.Stderr, "query is required")
		os.Exit(2)
	}
	if *maxResults <= 0 {
		*maxResults = 1
	}

	results, err := searchDuckDuckGo(*query, *maxResults, *timeoutSec)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	encoder := json.NewEncoder(os.Stdout)
	if err := encoder.Encode(results); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
