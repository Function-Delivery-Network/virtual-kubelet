package main

import (
	"github.com/ansjin/virtual-kubelet/cmd/virtual-kubelet/internal/provider"
	"github.com/ansjin/virtual-kubelet/cmd/virtual-kubelet/internal/provider/mock"
	"github.com/ansjin/virtual-kubelet/cmd/virtual-kubelet/internal/provider/fdn"
)

func registerMock(s *provider.Store) {
	s.Register("mock", func(cfg provider.InitConfig) (provider.Provider, error) { //nolint:errcheck
		return mock.NewMockProvider(
			cfg.ConfigPath,
			cfg.NodeName,
			cfg.OperatingSystem,
			cfg.InternalIP,
			cfg.DaemonPort,
		)
	})
	s.Register("fdn", func(cfg provider.InitConfig) (provider.Provider, error) { //nolint:errcheck
		return fdn.NewFDNProvider(
			cfg.ConfigPath,
			cfg.NodeName,
			cfg.OperatingSystem,
			cfg.InternalIP,
			cfg.DaemonPort,
		)
	})
}
