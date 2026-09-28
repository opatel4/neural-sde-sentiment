import math
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import GenerateGaussLaguerre
from in2_adapter import make_in2_batch
from pricers import (
    Hestonprice_Batched,
    Batesprice_Batched,
    Bergomiprice_Batched,
    rBergomiprice_Batched,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealWorldFineTuner:
    def __init__(
        self,
        model,
        x_mean,
        x_std,
        device,
        model_config,
        lm_convention='f_over_k',
        train_mc_settings=None,
        eval_mc_settings=None
    ):
        self.model = model
        self.device = device
        self.x_mean = x_mean.to(device)
        self.x_std = x_std.to(device).clamp_min(1e-7)
        self.model_config = model_config
        self.model_type = model_config["model_type"]
        self.lm_convention = lm_convention

        self.maturities = torch.tensor(
            [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0],
            dtype=torch.float32,
            device=device
        )
        self.log_moneyness = torch.tensor(
            [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            dtype=torch.float32,
            device=device
        )

        self.S = torch.tensor(1.0, dtype=torch.float32, device=device)
        # In2 wiring (env-controlled; defaults off = old In1 path)
        self._use_in2  = os.environ.get('SDP_USE_IN2', '0') == '1'
        self._use_sent = os.environ.get('SDP_USE_SENTIMENT', '1') == '1'

        # PATCHED: the old hardcoded 0.045/0.011 gives forward drift 0.034,
        # while data_utils normalises by SpotProxy built from the CSV's own r,q
        # (market drift 0.008-0.017). That mismatch mis-places every strike and
        # accounts for ~40-45% of the pricing error. Defaults below match the
        # market; override per regime with SDP_R / SDP_Q env vars.
        _r_default = float(os.environ.get("SDP_R", 0.0231))
        _q_default = float(os.environ.get("SDP_Q", 0.0150))
        self.r = torch.tensor(_r_default, dtype=torch.float32, device=device)
        self.q = torch.tensor(_q_default, dtype=torch.float32, device=device)

        self.x_nodes, self.w_nodes = GenerateGaussLaguerre(32, device=device)
        self.abs_criterion = nn.HuberLoss(delta=0.05)

        default_train_mc_settings = {
            "Heston":   {"N_paths": 3000, "N_steps": 100},
            "Bates":    {"N_paths": 3000, "N_steps": 100},
            "Bergomi":  {"N_paths": 3000, "N_steps": 100},
            "rBergomi": {"N_paths": 3000, "N_steps": 100},
        }

        self.train_mc_settings = (
            copy.deepcopy(train_mc_settings)
            if train_mc_settings is not None
            else default_train_mc_settings
        )
        self.eval_mc_settings = (
            copy.deepcopy(eval_mc_settings)
            if eval_mc_settings is not None
            else copy.deepcopy(self.train_mc_settings)
        )

    def scale_input(self, market_iv_surface):
        scaled_input = (market_iv_surface - self.x_mean) / self.x_std
        return scaled_input.clamp(-8.0, 8.0).to(torch.float32)

    def get_mc_settings(self, for_eval=False):
        settings_source = self.eval_mc_settings if for_eval else self.train_mc_settings
        if self.model_type not in settings_source:
            raise KeyError(f"Missing MC settings for model type: {self.model_type}")
        return settings_source[self.model_type]

    def price_one_maturity(self, params, T, for_eval=False):
        Fwd = self.S * torch.exp((self.r - self.q) * T)

        strikes = []
        for lm in self.log_moneyness:
            if self.lm_convention == 'k_over_f':
                K = Fwd * torch.exp(lm)
            else:
                K = Fwd * torch.exp(-lm)
            strikes.append(K)

        K_vec = torch.stack(strikes)
        mc = self.get_mc_settings(for_eval=for_eval)

        if self.model_type == "Bates":
            return Batesprice_Batched(
                params[:, 0], params[:, 1], params[:, 2], params[:, 3],
                params[:, 4], params[:, 5], params[:, 6], params[:, 7],
                T, K_vec, self.S, self.r, self.q,
                self.x_nodes, self.w_nodes,
                N_paths=mc["N_paths"], N_steps=mc["N_steps"]
            )
        elif self.model_type == "Heston":
            return Hestonprice_Batched(
                params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4],
                T, K_vec, self.S, self.r, self.q,
                self.x_nodes, self.w_nodes,
                N_paths=mc["N_paths"], N_steps=mc["N_steps"]
            )
        elif self.model_type == "Bergomi":
            return Bergomiprice_Batched(
                params[:, 0], params[:, 1], params[:, 2], params[:, 3],
                T, K_vec, self.S, self.r, self.q,
                N_paths=mc["N_paths"], N_steps=mc["N_steps"]
            )
        elif self.model_type == "rBergomi":
            return rBergomiprice_Batched(
                params[:, 0], params[:, 1], params[:, 2], params[:, 3],
                T, K_vec, self.S, self.r, self.q,
                N_paths=mc["N_paths"], N_steps=mc["N_steps"]
            )
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} is not wired into the fine tuner yet."
            )

    def calculate_model_price_surface(self, params, for_eval=False):
        model_prices = []
        for T in self.maturities:
            row_prices = self.price_one_maturity(params, T, for_eval=for_eval)
            model_prices.append(row_prices.unsqueeze(1))
        surface = torch.cat(model_prices, dim=1).unsqueeze(1)
        return surface.to(torch.float32)

    def bs_vega(self, market_iv_surface):
        sigma = market_iv_surface.clamp(min=1e-4)
        T = self.maturities.view(1, 1, -1, 1)
        k = self.log_moneyness.view(1, 1, 1, -1)
        sqrt_T = T.clamp(min=1e-8).sqrt()
        d1 = (-k + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        phi = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * d1**2)
        vega = self.S * torch.exp(-self.q * T) * phi * sqrt_T
        return vega.detach()

    def vega_weighted_loss(self, pred, target, market_iv, delta=0.05):
        vega = self.bs_vega(market_iv)
        w = vega / vega.mean().clamp(min=1e-10)
        huber = F.huber_loss(pred, target, reduction='none', delta=delta)
        return (w * huber).mean()

    def confirm_gradients(self, train_loader, param_names=None):
        if param_names is None:
            pb = self.model_config.get("param_bounds", [])
            if isinstance(pb, dict):
                param_names = list(pb.keys())
            else:
                param_names = [f"param_{i}" for i in range(len(pb))]

        self.model.train()
        self.model.zero_grad()
        batch = next(iter(train_loader))
        market_iv = batch[0].to(self.device, dtype=torch.float32)
        market_price = batch[1].to(self.device, dtype=torch.float32)
        context = batch[2].to(self.device, dtype=torch.float32)

        scaled_input = self.scale_input(market_iv)
        if getattr(self, '_use_in2', False):
            _x = make_in2_batch(scaled_input, context,
                                use_sentiment=getattr(self, '_use_sent', True))
            _, predicted_params = self.model(_x)
        else:
            _, predicted_params = self.model(scaled_input, context)
        predicted_params.retain_grad()
        surf = self.calculate_model_price_surface(predicted_params, for_eval=False)
        loss = self.vega_weighted_loss(surf, market_price, market_iv)
        loss.backward()

        print(f"\n=== Gradient audit [{self.model_type}] ===")
        dead = []
        n_cols = predicted_params.shape[1]
        for i in range(n_cols):
            name = param_names[i] if i < len(param_names) else f"param_{i}"
            mg = predicted_params.grad[:, i].abs().mean().item() \
                 if predicted_params.grad is not None else 0.0
            tag = " ~DEAD" if mg < 1e-9 else ""
            print(f"  {name}: mean|grad|={mg:.3e}{tag}")
            if mg < 1e-9:
                dead.append(name)
        total_wgrad = sum(
            p.grad.abs().sum().item()
            for p in self.model.parameters() if p.grad is not None
        )
        print(f"  Total |grad| in network weights: {total_wgrad:.3e}")
        print(f"  Dead columns: {dead if dead else 'none'}")
        return dead

    def relative_metric(self, pred_surface, target_surface):
        denom = target_surface.abs().clamp(min=0.05)
        return torch.mean(torch.abs(pred_surface - target_surface) / denom)

    def evaluate_loader(self, data_loader):
        if data_loader is None:
            return None

        self.model.eval()
        abs_loss_total = 0.0
        rel_loss_total = 0.0

        with torch.no_grad():
            for market_iv_surface, market_price_surface, context in data_loader:
                market_iv_surface = market_iv_surface.to(self.device, dtype=torch.float32)
                market_price_surface = market_price_surface.to(self.device, dtype=torch.float32)
                context = context.to(self.device, dtype=torch.float32)

                scaled_input = self.scale_input(market_iv_surface)
                if getattr(self, '_use_in2', False):
                    _x = make_in2_batch(scaled_input, context,
                                        use_sentiment=getattr(self, '_use_sent', True))
                    _, predicted_params = self.model(_x)
                else:
                    _, predicted_params = self.model(scaled_input, context)
                model_generated_price_surface = self.calculate_model_price_surface(
                    predicted_params, for_eval=True
                )

                abs_loss_total += self.vega_weighted_loss(
                    model_generated_price_surface, market_price_surface, market_iv_surface
                ).item()
                rel_loss_total += self.relative_metric(
                    model_generated_price_surface, market_price_surface
                ).item()

        n_batches = len(data_loader)
        return {
            "abs": abs_loss_total / n_batches,
            "rel": rel_loss_total / n_batches
        }

    def fine_tune(
        self,
        train_loader,
        val_loader=None,
        epochs=80,
        lr=5e-6,
        weight_decay=1e-5,
        scheduler_start_epoch=15,
        scheduler_patience=10,
        scheduler_factor=0.7,
        scheduler_cooldown=2,
        scheduler_min_lr=1e-6,
        best_model_path=None
    ):
        if best_model_path is None:
            best_model_dir = os.path.join(PROJECT_ROOT, 'sentiment_output')
            best_model_stem = 'Best_FineTuned_Model'
        else:
            best_model_dir = os.path.dirname(best_model_path)
            best_model_stem = os.path.splitext(os.path.basename(best_model_path))[0]
        best_model_path = None  # will be set dynamically per best epoch

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            cooldown=scheduler_cooldown,
            min_lr=scheduler_min_lr
        )

        history = {
            "train_abs": [],
            "train_rel": [],
            "val_abs": [],
            "val_rel": [],
            "lr": []
        }

        printed_debug = False
        best_loss = float('inf')
        best_epoch = -1
        prev_monitored_loss = None
        increase_streak = 0
        # V3: 10 consecutive rises fired before the LR scheduler even starts
        # (scheduler_start_epoch=15). In the 2013-2015 regime EVERY cell's
        # loss rises through the first ~10 epochs; rBergomi survived only
        # because one non-monotonic step reset the counter, then went on to
        # the best loss in that regime (0.000781). Raised to 30 and not
        # armed until after the scheduler engages.
        max_increase_streak = 30
        early_stop_arm_epoch = 20

        train_days = len(train_loader.dataset)
        val_days = 0 if val_loader is None else len(val_loader.dataset)

        tqdm.write(f"Starting Fine-Tuning for {self.model_type} on {train_days} market days...")
        if val_loader is None:
            tqdm.write("Validation disabled because dataset is too small for a stable split.")
        else:
            tqdm.write(f"Validation days: {val_days}")
        tqdm.write(
            f"MC settings (train) for {self.model_type}: "
            f"{self.train_mc_settings[self.model_type]}"
        )
        tqdm.write(
            f"MC settings (eval)  for {self.model_type}: "
            f"{self.eval_mc_settings[self.model_type]}"
        )
        tqdm.write(
            f"Scheduler -> start: {scheduler_start_epoch}, patience: {scheduler_patience}, "
            f"factor: {scheduler_factor}, cooldown: {scheduler_cooldown}, "
            f"min_lr: {scheduler_min_lr}"
        )

        dead_params = self.confirm_gradients(train_loader)
        tqdm.write(f"  confirm_gradients dead: {dead_params if dead_params else 'none'}")

        epoch_bar = tqdm(range(epochs), desc=f"{self.model_type} Fine-Tune", unit="epoch")

        for epoch in epoch_bar:
            self.model.train()
            train_abs = 0.0
            train_rel = 0.0

            batch_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:03d}/{epochs}",
                unit="batch",
                leave=False
            )
            for market_iv_surface, market_price_surface, context in batch_bar:
                market_iv_surface = market_iv_surface.to(self.device, dtype=torch.float32)
                market_price_surface = market_price_surface.to(self.device, dtype=torch.float32)
                context = context.to(self.device, dtype=torch.float32)

                scaled_input = self.scale_input(market_iv_surface)

                optimizer.zero_grad()
                if getattr(self, '_use_in2', False):
                    _x = make_in2_batch(scaled_input, context,
                                        use_sentiment=getattr(self, '_use_sent', True))
                    _, predicted_params = self.model(_x)
                else:
                    _, predicted_params = self.model(scaled_input, context)
                model_generated_price_surface = self.calculate_model_price_surface(
                    predicted_params, for_eval=False
                )

                if not printed_debug:
                    tqdm.write(
                        f"  scaled IV min/max: "
                        f"{scaled_input.min().item():.4f} / {scaled_input.max().item():.4f}"
                    )
                    tqdm.write(
                        f"  market min/max:    "
                        f"{market_price_surface.min().item():.6f} / "
                        f"{market_price_surface.max().item():.6f}"
                    )
                    tqdm.write(
                        f"  model  min/max:    "
                        f"{model_generated_price_surface.min().item():.6f} / "
                        f"{model_generated_price_surface.max().item():.6f}"
                    )
                    printed_debug = True

                abs_batch_loss = self.vega_weighted_loss(
                    model_generated_price_surface, market_price_surface, market_iv_surface
                )
                rel_batch_loss = self.relative_metric(
                    model_generated_price_surface, market_price_surface
                )

                abs_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_abs += abs_batch_loss.item()
                train_rel += rel_batch_loss.item()
                batch_bar.set_postfix(
                    huber=f"{abs_batch_loss.item():.6f}",
                    rel=f"{rel_batch_loss.item():.6f}"
                )

            avg_train_abs = train_abs / len(train_loader)
            avg_train_rel = train_rel / len(train_loader)

            val_metrics = self.evaluate_loader(val_loader)

            if val_metrics is None:
                monitored_loss = avg_train_abs
                current_val_abs = None
                current_val_rel = None
            else:
                monitored_loss = val_metrics["abs"]
                current_val_abs = val_metrics["abs"]
                current_val_rel = val_metrics["rel"]

            if prev_monitored_loss is not None and monitored_loss >= prev_monitored_loss:
                increase_streak += 1
            else:
                increase_streak = 0
            prev_monitored_loss = monitored_loss

            if epoch + 1 >= early_stop_arm_epoch and increase_streak >= max_increase_streak:
                tqdm.write(
                    f"  Early stopping at epoch {epoch+1}: loss increased for "
                    f"{max_increase_streak} consecutive epochs."
                )
                break

            if epoch + 1 >= scheduler_start_epoch:
                scheduler.step(monitored_loss)

            current_lr = optimizer.param_groups[0]['lr']

            history["train_abs"].append(avg_train_abs)
            history["train_rel"].append(avg_train_rel)
            history["val_abs"].append(current_val_abs)
            history["val_rel"].append(current_val_rel)
            history["lr"].append(current_lr)

            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_epoch = epoch + 1

                # Remove previous best file before saving new one
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                best_model_path = os.path.join(
                    best_model_dir, f"{best_model_stem}_epoch_{best_epoch}.pth"
                )

                best_payload = {
                    "epoch": best_epoch,
                    "best_loss": best_loss,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "x_mean": self.x_mean.detach().cpu(),
                    "x_std": self.x_std.detach().cpu(),
                    "model_type": self.model_type,
                    "train_mc_settings": self.train_mc_settings,
                    "eval_mc_settings": self.eval_mc_settings
                }

                torch.save(best_payload, best_model_path)

            if val_metrics is None:
                epoch_bar.set_postfix(
                    huber=f"{avg_train_abs:.6f}",
                    rel=f"{avg_train_rel:.6f}",
                    lr=f"{current_lr:.2e}",
                    best=f"{best_epoch}"
                )
            else:
                epoch_bar.set_postfix(
                    tr_huber=f"{avg_train_abs:.6f}",
                    val_huber=f"{current_val_abs:.6f}",
                    tr_rel=f"{avg_train_rel:.6f}",
                    val_rel=f"{current_val_rel:.6f}",
                    lr=f"{current_lr:.2e}",
                    best=f"{best_epoch}"
                )

        # Save final epoch with full history
        final_epoch = epoch + 1
        final_path = os.path.join(
            best_model_dir, f"{best_model_stem}_final_epoch_{final_epoch}.pth"
        )
        final_payload = {
            "epoch": final_epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "x_mean": self.x_mean.detach().cpu(),
            "x_std": self.x_std.detach().cpu(),
            "model_type": self.model_type,
            "train_mc_settings": self.train_mc_settings,
            "eval_mc_settings": self.eval_mc_settings
        }
        torch.save(final_payload, final_path)

        tqdm.write(
            f"Training complete. Best epoch: {best_epoch} | Best loss: {best_loss:.6f}"
        )
        return history, best_epoch, best_loss
