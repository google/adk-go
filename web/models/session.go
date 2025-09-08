package models

import (
	"time"
)

type Session struct {
	ID        string         `json:"id"`
	AppName   string         `json:"app_name"`
	UserID    string         `json:"user_id"`
	UpdatedAt time.Time      `json:"updated_at"`
	Events    []Event        `json:"events"`
	State     map[string]any `json:"state"`
}
