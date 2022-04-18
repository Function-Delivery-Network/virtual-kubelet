package main

import (
	"github.com/Function-Delivery-Network/virtual-kubelet/cmd/virtual-kubelet/internal/provider"
	"github.com/Function-Delivery-Network/virtual-kubelet/cmd/virtual-kubelet/internal/provider/fdn"
	"github.com/Function-Delivery-Network/virtual-kubelet/cmd/virtual-kubelet/internal/provider/mock"
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
			cfg.ServerlessPlatformName,
			cfg.ServerlessPlatformApiHost,
			cfg.ServerlessPlatformAuth,
			cfg.MinioEndpoint,
			cfg.MinioAccessKeyID, 
			cfg.MinioSecretAccessKey,
		)
	})
}
